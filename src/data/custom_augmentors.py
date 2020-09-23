import torch
from fairseq import utils
import copy
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np


class BatchBackTranslationAug:

    def __init__(self, model_names, from_num_beam=5, to_num_beam=5, device='cuda:0'):

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

    def batch_augments(self, sentences, batch_size=50):
        self.from_model.eval()
        self.to_model.eval()

        result = []
        oom = False

        try:
            for batch_ind in tqdm(range(len(sentences)//batch_size + 1)):
                inputs = [self.from_model.encode(sample) for sample in sentences[batch_ind*batch_size:(batch_ind+1)*batch_size]]

                dataset = self.from_model.task.build_dataset_for_inference(inputs, [input.numel() for input in inputs])
                sample = dataset.collater(dataset)
                sample = utils.apply_to_sample(lambda tensor: tensor.to(self.from_model.device), sample)
                gen_args = copy.copy(self.from_model.args)
                gen_args.beam = self.from_num_beam
                generator = self.from_model.task.build_generator(gen_args)
                translations = self.from_model.task.inference_step(generator, self.from_model.models, sample)
                translations = [self.from_model.decode(tr[0]['tokens']) for tr in translations]
                translations = [translations[sample['id'].tolist().index(i)] for i in range(len(translations))]

                translations = [self.to_model.encode(sample) for sample in translations]
                dataset = self.to_model.task.build_dataset_for_inference(translations, [input.numel() for input in translations])
                sample = dataset.collater(dataset)
                sample = utils.apply_to_sample(lambda tensor: tensor.to(self.to_model.device), sample)
                gen_args = copy.copy(self.to_model.args)
                gen_args.beam = self.to_num_beam
                generator = self.to_model.task.build_generator(gen_args)
                back_translations = self.to_model.task.inference_step(generator, self.to_model.models, sample)
                back_translations = [self.to_model.decode(tr[0]['tokens']) for tr in back_translations]
                back_translations = [back_translations[sample['id'].tolist().index(i)] for i in range(len(back_translations))]

                result.extend(back_translations)

        except RuntimeError:
            torch.cuda.empty_cache()
            gc.collect()
            oom = True

        if oom:
            return self.batch_augments(sentences, batch_size=batch_size//2)

        return result


class BatchAbstSummAug:

    def __init__(self, model_path, max_length, num_beam=3, device='cuda:1'):

        self.num_beam = num_beam
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        self.text_prefix = 'summarize: '
        self.model.to(device)

    def batch_augments(self, sentences, batch_size=25):
        self.model.eval()
        order = np.argsort([len(sent) for sent in sentences])
        sorted_sentences = np.array(sentences)[order]

        result = []
        oom = False

        try:
            for batch_ind in tqdm(range(len(sorted_sentences)//batch_size + 1)):
                sentence_batch = sorted_sentences[batch_ind*batch_size:(batch_ind+1)*batch_size]
                max_len = max(int(self.max_length * min([len(sent) for sent in sentence_batch])), 10)

                token_ids = self.tokenizer([self.text_prefix + sent for sent in sentence_batch], return_tensors='pt', padding=True)
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
            return self.batch_augments(sentences, batch_size=batch_size//2)

        return result
