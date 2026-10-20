from __future__ import annotations
import torch
from .preprocessing import normalize, entity_signature

class Reformulator:
    def __init__(self, name, cache_dir="models", rank=4, alpha=8, language="te", device="cpu"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        # This checkpoint explicitly documents the slow AlbertTokenizer path. `use_fast=True`
        # attempts an unsupported conversion and fails when its legacy vocab_file is None.
        self.tokenizer=AutoTokenizer.from_pretrained(
            name, cache_dir=cache_dir, use_fast=False,
            do_lower_case=False, keep_accents=True,
        )
        self.device=torch.device(device)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(name,cache_dir=cache_dir).to(self.device)
        self.model.train()
        self.base_name=name
        self.language=language
        self.target_lang_token=f"<2{language}>"
        self.target_lang_id=self.tokenizer._convert_token_to_id_with_added_voc(self.target_lang_token)
        if self.target_lang_id == self.tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer does not contain required language token {self.target_lang_token}")
        self.model.config.decoder_start_token_id=self.target_lang_id
        self.model.generation_config.decoder_start_token_id=self.target_lang_id
        self.cache={}
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            modules=set(n.split(".")[-1] for n,_ in self.model.named_modules())
            targets=[m for m in ("q","v","k","o") if m in modules]
            if not targets: targets=[m for m in ("q_proj","v_proj") if m in modules]
            if targets:
                self.model=get_peft_model(self.model,LoraConfig(r=rank,lora_alpha=alpha,target_modules=targets,task_type=TaskType.SEQ_2_SEQ_LM)); self.peft=True
            else: raise RuntimeError("No compatible attention projection modules for LoRA")
        except Exception as e:
            raise RuntimeError(f"Could not attach LoRA adapter to {name}: {e}") from e
        self.trainable=[p for p in self.model.parameters() if p.requires_grad]
        self.trainable_count=sum(p.numel() for p in self.trainable)

    def _source(self, question, max_input):
        text=f"{normalize(question)} </s> {self.target_lang_token}"
        encoded=self.tokenizer(text,add_special_tokens=False,return_tensors="pt",truncation=True,max_length=max_input)
        # AlbertTokenizer can emit token_type_ids even though MBART encoder-decoder
        # generation does not accept them.
        return {key:value.to(self.device) for key,value in encoded.items() if key in {"input_ids","attention_mask"}}

    def sample(self, question, n=2, max_input=96, max_new=24, do_sample=True):
        """Return display candidates, replacing invalid outputs with the input question."""
        return [x["candidate"] for x in self.sample_actions(question, n, max_input, max_new, do_sample)]

    def sample_actions(self, question, n=2, max_input=96, max_new=24, do_sample=True):
        """Sample policy actions while retaining the exact generated token IDs.

        Invalid text is exposed as the original question for downstream answer scoring,
        but the original sampled IDs remain attached for the policy gradient. This keeps
        the REINFORCE likelihood paired with the actual sampled action.
        """
        inp=self._source(question,max_input)
        was_training=self.model.training
        self.model.eval()
        generation_kwargs={"max_new_tokens":max_new,"do_sample":do_sample,
                           "num_beams":max(1,n) if not do_sample else 1,
                           "num_return_sequences":n,
                           "pad_token_id":self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                           "decoder_start_token_id":self.target_lang_id}
        if do_sample:
            generation_kwargs.update(top_p=.9,temperature=.9)
        with torch.inference_mode():
            seq=self.model.generate(**inp,**generation_kwargs)
        self.model.train(was_training)
        actions=[]
        for ids in seq:
            text=normalize(self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
            tokens=ids.tolist()
            if tokens and tokens[0] == self.target_lang_id:
                tokens=tokens[1:]
            if self.tokenizer.eos_token_id in tokens:
                tokens=tokens[:tokens.index(self.tokenizer.eos_token_id)+1]
            else:
                tokens=[x for x in tokens if x != self.tokenizer.pad_token_id]
            valid=self._valid(question,text)
            actions.append({"sampled_text":text,"candidate":text if valid else question,
                            "valid":valid,"token_ids":tokens})
        return actions

    def _valid(self, original, candidate):
        if not candidate or len(candidate)>max(300,4*len(original)): return False
        if len(candidate)<2 or candidate==original: return False
        return entity_signature(original).issubset(entity_signature(candidate))

    def sequence_logprob(self, question, sampled, max_input=96, max_new=24):
        """Teacher-force the exact generated action (string convenience wrapper)."""
        tgt=self.tokenizer(normalize(sampled),add_special_tokens=False,return_tensors="pt",truncation=True,max_length=max(1,max_new-1))
        ids=tgt.input_ids[0].tolist()
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return self.sequence_logprob_ids(question,ids,max_input)

    def sequence_logprob_ids(self, question, token_ids, max_input=96):
        if not token_ids:
            raise ValueError("A sampled reformulation must contain at least one generated token")
        src=self._source(question,max_input)
        # `generate` returns decoder-start followed by sampled target IDs. Its start
        # token is omitted here; teacher forcing adds it via the model's shift-right.
        labels=torch.tensor([token_ids],dtype=torch.long,device=self.device)
        out=self.model(**src,labels=labels)
        logp=torch.log_softmax(out.logits,dim=-1)
        selected=logp.gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        return selected.sum()

    def save_adapter(self,path): self.model.save_pretrained(path)
