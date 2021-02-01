import torch
from fairseq import utils
import copy
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from nlpaug.model.lang_models import XlNet, Gpt2


class BatchBackTranslationAug:

    def __init__(self, model_names, from_num_beam=5, to_num_beam=5, device='cuda'):

        self.from_num_beam = from_num_beam
        self.to_num_beam = to_num_beam
        self.from_model = torch.hub.load(github='pytorch/fairseq', model=model_names[0],
                                         checkpoint_file='model1.pt',
                                         tokenizer='moses', bpe='fastbpe')

        self.to_model = torch.hub.load(github='pytorch/fairseq', model=model_names[1],
                                       checkpoint_file='model1.pt',
                                       tokenizer='moses', bpe='fastbpe')

        self.from_model.cuda(device=device)
        self.to_model.cuda(device=device)

    def batch_augments(self, sentences, batch_size=30, progress_bar=True):
        self.from_model.eval()
        self.to_model.eval()

        result = []
        oom = False
        batch_ind = 0

        iterator = tqdm(range(len(sentences) // batch_size + 1)) if progress_bar else range(len(sentences) // batch_size + 1)

        try:
            for batch_ind in iterator:
                inputs = [self.from_model.encode(sample) for sample in
                          sentences[batch_ind * batch_size:(batch_ind + 1) * batch_size]]

                if len(inputs) > 0:

                    dataset = self.from_model.task.build_dataset_for_inference(inputs, [input.numel() for input in inputs])
                    sample = dataset.collater(dataset)
                    sample = utils.apply_to_sample(lambda tensor: tensor.to(self.from_model.device), sample)
                    gen_args = copy.copy(self.from_model.args)
                    gen_args.beam = self.from_num_beam
                    generator = self.from_model.task.build_generator(self.from_model.models, args=gen_args)
                    translations = self.from_model.task.inference_step(generator, self.from_model.models, sample)
                    translations = [self.from_model.decode(tr[0]['tokens']) for tr in translations]
                    translations = [translations[sample['id'].tolist().index(i)] for i in range(len(translations))]

                    translations = [self.to_model.encode(sample) for sample in translations]
                    dataset = self.to_model.task.build_dataset_for_inference(translations, [input.numel() for input in translations])
                    sample = dataset.collater(dataset)
                    sample = utils.apply_to_sample(lambda tensor: tensor.to(self.to_model.device), sample)
                    gen_args = copy.copy(self.to_model.args)
                    gen_args.beam = self.to_num_beam
                    generator = self.to_model.task.build_generator(self.to_model.models, args=gen_args)
                    back_translations = self.to_model.task.inference_step(generator, self.to_model.models, sample)
                    back_translations = [self.to_model.decode(tr[0]['tokens']) for tr in back_translations]
                    back_translations = [back_translations[sample['id'].tolist().index(i)] for i in range(len(back_translations))]

                    result.extend(back_translations)

        except RuntimeError:
            torch.cuda.empty_cache()
            gc.collect()
            oom = True

        if oom:
            result.extend(self.batch_augments(sentences[batch_ind * batch_size:(batch_ind + 1) * batch_size],
                                              batch_size=batch_size // 2, progress_bar=False))

            result.extend(self.batch_augments(sentences[(batch_ind + 1) * batch_size:],
                                              batch_size=batch_size))

        return result


class BatchAbstSummAug:

    def __init__(self, model_path, max_length, num_beam=100, device='cuda'):

        self.num_beam = num_beam
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.text_prefix = 'summarize: '
        self.model.to(device)

    def batch_augments(self, sentences, batch_size=30, progress_bar=True):
        self.model.eval()
        order = np.argsort([len(sent) for sent in sentences])
        sorted_sentences = np.array(sentences)[order]

        result = []
        oom = False
        batch_ind = 0

        iterator = tqdm(range(len(sorted_sentences) // batch_size + 1)) if progress_bar \
            else range(len(sorted_sentences) // batch_size + 1)

        try:
            for batch_ind in iterator:
                sentence_batch = sorted_sentences[batch_ind * batch_size:(batch_ind + 1) * batch_size]
                if len(sentence_batch) > 0:
                    max_len = max(int(self.max_length * min([len(sent) for sent in sentence_batch])), 10)

                    token_ids = self.tokenizer([self.text_prefix + sent for sent in sentence_batch], return_tensors='pt',
                                               padding=True)
                    # left truncation
                    token_ids['input_ids'] = token_ids['input_ids'].to(self.model.device)

                    target_token_ids = self.model.generate(token_ids['input_ids'], min_length=10,
                                                           max_length=max_len, num_beams=self.num_beam, no_repeat_ngram_size=3)

                    summarizations = [self.tokenizer.decode(target_token_id, skip_special_tokens=True)
                                      for target_token_id in target_token_ids]

                    result.extend(summarizations)

            result = [result[list(order).index(i)] for i in range(len(result))]

        except RuntimeError:
            torch.cuda.empty_cache()
            gc.collect()
            oom = True

        if oom:
            result.extend(self.batch_augments(sorted_sentences[batch_ind * batch_size:(batch_ind + 1) * batch_size],
                                              batch_size=batch_size // 2, progress_bar=False))

            result.extend(self.batch_augments(sorted_sentences[(batch_ind + 1) * batch_size:],
                                              batch_size=batch_size))

            result = [result[list(order).index(i)] for i in range(len(result))]

        return result


def predict(self, texts, target_words=None, n=1, external_memory=None,
            include_punctuation=False):
    # Prepare inputs
    input_idxes = [self.tokenizer.encode(text)[-1024:] for text in texts]
    if target_words is None:
        target_words = [None] * len(input_idxes)
        # target_words = [t.replace(self.SUBWORD_PREFIX, '') for t in target_words if t]

    # Pad token
    max_token_size = max([len(t) for t in input_idxes])
    for i, token_input in enumerate(input_idxes):
        for _ in range(max_token_size - len(token_input)):
            input_idxes[i].append(self.pad_id)

    target_poses = []
    if external_memory is None:  # First step or does not enable optimization
        for i, tokens in enumerate(input_idxes):
            target_poses.append(len(self.padding_text_idxes) + tokens.index(self.mask_id))
            input_idxes[i] = self.padding_text_idxes + tokens
    else:
        for i, tokens in enumerate(input_idxes):
            target_poses.append(tokens.index(self.mask_id))

    perm_masks = torch.zeros((len(input_idxes), len(input_idxes[0]), len(input_idxes[0])), dtype=torch.float)
    target_mappings = torch.zeros((len(input_idxes), 1, len(input_idxes[0])), dtype=torch.float)
    for i, target_pos in enumerate(target_poses):
        perm_masks[i][:, target_pos] = 1.0  # Mask the target word
        target_mappings[i, 0, target_pos] = 1.0

    # Convert to feature
    input_idxes = torch.tensor(input_idxes).to(self.device)
    perm_masks = perm_masks.to(self.device)
    target_mappings = target_mappings.to(self.device)

    # Prediction
    results = []
    with torch.no_grad():
        outputs = self.model(input_ids=input_idxes, perm_mask=perm_masks, target_mapping=target_mappings,
                             mems=external_memory)

    # Selection
    for output, target_token in zip(outputs[0], target_words):
        target_token_logits = output[0]

        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
        if len(target_token_idxes) != 0:
            new_tokens = self.pick(target_token_logits, target_token_idxes, target_word=target_token,
                                   n=10, include_punctuation=include_punctuation)
            results.append([t[0] for t in new_tokens])
        else:
            results.append([''])

    return results


XlNet.predict = predict


def predict(self, texts, target_words=None, n=1, external_memory=None,
            include_punctuation=False):
    # Prepare inputs
    input_idxes = [self.tokenizer.encode(text)[-1024:] for text in texts]
    if target_words is None:
        target_words = [None] * len(input_idxes)
    mask_inputs = []

    # Pad token
    max_token_size = max([len(t) for t in input_idxes])
    for i, token_input in enumerate(input_idxes):
        mask_input = [1] * len(input_idxes[0])  # 1: are not masked, 0: masked token (for padding)

        for _ in range(max_token_size - len(token_input)):
            input_idxes[i].append(self.pad_id)
            mask_input.append(0)

        mask_inputs.append(mask_input)

    # Convert to feature
    input_idxes = torch.tensor(input_idxes).to(self.device)
    mask_inputs = torch.tensor(mask_inputs).to(self.device)

    # Prediction
    results = []
    with torch.no_grad():
        outputs = self.model(input_ids=input_idxes, attention_mask=mask_inputs, past_key_values=external_memory)

    # Selection
    for output, target_token in zip(outputs[0], target_words):
        target_token_logits = output[0]

        seed = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p}
        target_token_logits = self.control_randomness(target_token_logits, seed)
        target_token_logits, target_token_idxes = self.filtering(target_token_logits, seed)
        if len(target_token_idxes) != 0:
            new_tokens = self.pick(target_token_logits, target_token_idxes, target_word=target_token,
                                   n=10, include_punctuation=include_punctuation)
            results.append([t[0] for t in new_tokens])
        else:
            results.append([''])

    return results


Gpt2.predict = predict
