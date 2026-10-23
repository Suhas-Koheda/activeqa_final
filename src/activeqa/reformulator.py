from __future__ import annotations
import re
import torch
from .preprocessing import normalize, entity_signature, interrogatives, tokens


_TELUGU_RE = re.compile(r"[\u0c00-\u0c7f]")
_FOREIGN_INDIC_RE = re.compile(r"[\u0900-\u0bff\u0d00-\u0d7f]")
_OUTPUT_MARKER_RE = re.compile(r"\[(?:more|continued|truncated)\]", re.I)
_TELUGU_FUNCTION_WORDS = {
    "ఏ", "ఏది", "ఏమి", "ఏమిటి", "ఎవరు", "ఎప్పుడు", "ఎక్కడ", "ఎందుకు", "ఎలా",
    "ఎన్ని", "ఎంత", "లో", "కి", "కు", "తో", "మరియు", "అని", "గా", "నే", "ను",
    "ది", "లు", "ఒక", "ఈ", "ఆ", "ఏదైనా", "చెప్పండి", "తెలుపుము", "పేర్కొనండి",
}


def _answer_type(text):
    words = set(tokens(text))
    if "శతాబ్దం" in words or "శతాబ్దంలో" in words or "శతాబ్దానికి" in words:
        return "century"
    if any(w.startswith("సంవత్సర") for w in words): return "year"
    if any(w.startswith("తేదీ") for w in words): return "date"
    if any(w.startswith("ఎన్ని") for w in words) or "సంఖ్య" in words: return "count"
    if "ఎంత" in words: return "amount"
    if any(w in words for w in ("ఎక్కడ", "ప్రదేశం", "నగరంలో", "స్థానం")): return "place"
    if "ఎందుకు" in words: return "reason"
    if any(w in words for w in ("ఎలా", "విధంగా")): return "manner"
    if any(w in words for w in ("ఎప్పుడు", "సమయంలో", "కాలంలో")): return "time"
    if any(w in words for w in ("ఎవరు", "ఎవరిని", "ఎవరి", "వ్యక్తి", "రచయిత", "ప్రధాని", "రాష్ట్రపతి")) or any(w.startswith("కళాకార") for w in words):
        return "person"
    if any(w in words for w in ("ఏది", "ఏమిటి", "ఏమి", "పేరు", "ఏ")): return "thing"
    return "unknown"


def _preserves_answer_type(source, candidate):
    src, dst = _answer_type(source), _answer_type(candidate)
    if src == dst and src != "unknown": return True
    # Telugu often changes an explicit “ఎవరు?” into a grammatical “వ్యక్తి పేరు ఏమిటి?”.
    if src == "person" and dst == "thing":
        return bool(re.search(r"పేరు|వ్యక్తి|రచయిత|కళాకార", candidate))
    return False


def _anchor_overlap(source, candidate):
    """Conservative lexical guard against dropping Telugu names/topic words."""
    source_words = [w for w in tokens(source) if len(w) >= 3 and w not in _TELUGU_FUNCTION_WORDS]
    target_words = tokens(candidate)
    if not source_words:
        return True
    # Names may acquire a Telugu case ending in a rewrite; allow a small edit
    # distance while requiring most source content words to remain recognizable.
    from difflib import SequenceMatcher
    matched = 0
    for word in source_words:
        if any(word == other or word in other or other in word or
               SequenceMatcher(None, word, other).ratio() >= .78
               for other in target_words):
            matched += 1
    return matched / len(source_words) >= .55

class Reformulator:
    def __init__(self, name, cache_dir="models", rank=None, alpha=None, language="te", device="cpu"):
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
        # The model card sets BOS/EOS explicitly; relying on a possibly incomplete
        # generation_config can make generate() run to max_new_tokens every time.
        self.bos_token_id=self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        self.eos_token_id=self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        if self.bos_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Tokenizer does not define the expected <s> start token")
        if self.eos_token_id == self.tokenizer.unk_token_id:
            self.eos_token_id=self.tokenizer.eos_token_id
        if self.eos_token_id is None or self.eos_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Tokenizer does not define the expected </s> end token")
        self.model.config.decoder_start_token_id=self.target_lang_id
        self.model.generation_config.decoder_start_token_id=self.target_lang_id
        self.model.config.bos_token_id=self.bos_token_id
        self.model.config.eos_token_id=self.eos_token_id
        self.model.generation_config.bos_token_id=self.bos_token_id
        self.model.generation_config.eos_token_id=self.eos_token_id
        self.cache={}
        # This is a full-parameter policy-gradient run: every reformulator
        # parameter is trainable. Gradient checkpointing keeps it practical on
        # Colab T4 GPUs without replacing the RL update with an adapter-only fit.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.trainable=[p for p in self.model.parameters() if p.requires_grad]
        self.trainable_count=sum(p.numel() for p in self.trainable)

    def _source(self, question, max_input):
        text=f"{normalize(question)} </s> {self.target_lang_token}"
        encoded=self.tokenizer(text,add_special_tokens=False,return_tensors="pt",truncation=True,max_length=max_input)
        # AlbertTokenizer can emit token_type_ids even though MBART encoder-decoder
        # generation does not accept them.
        return {key:value.to(self.device) for key,value in encoded.items() if key in {"input_ids","attention_mask"}}

    def sample(self, question, n=2, max_input=96, max_new=64, do_sample=True):
        """Return display candidates, replacing invalid outputs with the input question."""
        return [x["candidate"] for x in self.sample_actions(question, n, max_input, max_new, do_sample)]

    def sample_actions(self, question, n=2, max_input=96, max_new=64, do_sample=True):
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
                           "bos_token_id":self.bos_token_id,
                           "eos_token_id":self.eos_token_id,
                           "pad_token_id":self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_token_id,
                           "decoder_start_token_id":self.target_lang_id}
        generation_kwargs.update(no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3)
        if do_sample:
            generation_kwargs.update(top_p=.9,temperature=.9)
        with torch.inference_mode():
            seq=self.model.generate(**inp,**generation_kwargs)
        self.model.train(was_training)
        actions=[]
        for ids in seq:
            text=normalize(self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
            tokens=ids.tolist()
            terminated = self.eos_token_id in tokens
            if tokens and tokens[0] == self.target_lang_id:
                tokens=tokens[1:]
            if self.eos_token_id in tokens:
                tokens=tokens[:tokens.index(self.eos_token_id)+1]
            else:
                tokens=[x for x in tokens if x != self.tokenizer.pad_token_id]
            question_type_preserved=_preserves_answer_type(question, text)
            valid=self._valid(question,text,terminated=terminated)
            actions.append({"sampled_text":text,"candidate":text if valid else question,
                            "valid":valid,"terminated":terminated,"token_ids":tokens,
                            "question_type_preserved":question_type_preserved})
        return actions

    def _valid(self, original, candidate, terminated=True):
        # Reaching max_new_tokens without EOS usually means a mid-sentence cut.
        if not terminated: return False
        if not candidate or len(candidate)>max(300,4*len(original)): return False
        if len(candidate)<2 or candidate==original: return False
        if _OUTPUT_MARKER_RE.search(candidate): return False
        # MultiIndic's shared tokenizer may emit Bengali/Devanagari fragments for
        # Telugu input. Do not let those actions reach the policy reward or demo.
        if _FOREIGN_INDIC_RE.search(candidate): return False
        # Reject outputs that have collapsed into Latin/tokenizer noise. Allow
        # Latin names and numbers when the actual question remains Telugu.
        telugu_letters = len(_TELUGU_RE.findall(candidate))
        if telugu_letters < 3 or telugu_letters / max(1, len(candidate)) < .20: return False
        # Preserve the answer type (which/who/when/where/why/how) as well as
        # numeric/Latin entities; changing the interrogative changes the answer.
        if not _preserves_answer_type(original, candidate): return False
        if not _anchor_overlap(original, candidate): return False
        return entity_signature(original).issubset(entity_signature(candidate))

    def sequence_logprob(self, question, sampled, max_input=96, max_new=64):
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
        # Score under the same (dropout-free) policy that generated the actions.
        was_training=self.model.training
        self.model.eval()
        try:
            out=self.model(**src,labels=labels)
        finally:
            self.model.train(was_training)
        logp=torch.log_softmax(out.logits,dim=-1)
        selected=logp.gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        return selected.sum()

    def sequence_mean_logprob(self, question, sampled, max_input=96, max_new=64):
        """Mean token log likelihood, for length-neutral supervised warm-up."""
        tgt=self.tokenizer(normalize(sampled),add_special_tokens=False,return_tensors="pt",
                           truncation=True,max_length=max(1,max_new-1))
        ids=tgt.input_ids[0].tolist()
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return self.sequence_logprob_ids(question, ids, max_input) / max(1, len(ids))

    def sequence_logprob_ids_batch(self, question, token_ids_list, max_input=96):
        """Teacher-force several sampled actions for one question in one pass.

        Returns a tensor of shape (len(token_ids_list),) with the summed
        log-probability of each sampled token sequence. Batching this step is the
        main training speedup: it replaces one full encoder-decoder forward per
        candidate with a single forward over all candidates.
        """
        if not token_ids_list:
            raise ValueError("At least one sampled reformulation is required")
        src=self._source(question,max_input)
        batch=len(token_ids_list)
        src={key:value.expand(batch,-1) for key,value in src.items()}
        max_len=max(len(ids) for ids in token_ids_list)
        labels=torch.full((batch,max_len),-100,dtype=torch.long,device=self.device)
        mask=torch.zeros((batch,max_len),dtype=torch.bool,device=self.device)
        for i,ids in enumerate(token_ids_list):
            if not ids:
                raise ValueError("A sampled reformulation must contain at least one token")
            labels[i,:len(ids)]=torch.tensor(ids,dtype=torch.long,device=self.device)
            mask[i,:len(ids)]=True
        # Score under the same (dropout-free) policy that generated the actions.
        was_training=self.model.training
        self.model.eval()
        try:
            out=self.model(**src,labels=labels)
        finally:
            self.model.train(was_training)
        logp=torch.log_softmax(out.logits,dim=-1)
        selected=logp.gather(-1,labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (selected*mask).sum(dim=1)

    def save_adapter(self,path): self.model.save_pretrained(path)
