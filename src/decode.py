"""
train
"""
import os
from functools import partial

import torch
from tqdm import tqdm

import dataloader
import model
import transformer
import util
from trainer import BaseTrainer
from decoding import Decode, get_decode_fn
from model import dummy_mask

tqdm.monitor_interval = 0

tqdm = partial(tqdm, bar_format="{l_bar}{r_bar}")


class Data(util.NamedEnum):
    g2p = "g2p"
    p2g = "p2g"
    p2word = "p2word"
    infp2word = "infp2word"
    news15 = "news15"
    histnorm = "histnorm"
    sigmorphon16task1 = "sigmorphon16task1"
    sigmorphon17task1 = "sigmorphon17task1"
    sigmorphon19task1 = "sigmorphon19task1"
    sigmorphon19task2 = "sigmorphon19task2"
    lemma = "lemma"
    lemmanotag = "lemmanotag"
    lematus = "lematus"
    unimorph = "unimorph"


class Arch(util.NamedEnum):
    soft = "soft"  # soft attention without input-feeding
    hard = "hard"  # hard attention with dynamic programming without input-feeding
    approxihard = "approxihard"  # hard attention with REINFORCE approximation without input-feeding
    softinputfeed = "softinputfeed"  # soft attention with input-feeding
    largesoftinputfeed = (
        "largesoftinputfeed"  # soft attention with uncontrolled input-feeding
    )
    approxihardinputfeed = "approxihardinputfeed"  # hard attention with REINFORCE approximation with input-feeding
    hardmono = "hardmono"  # hard monotonic attention
    hmm = "hmm"  # 0th-order hard attention without input-feeding
    hmmfull = "hmmfull"  # 1st-order hard attention without input-feeding
    transformer = "transformer"
    universaltransformer = "universaltransformer"
    tagtransformer = "tagtransformer"
    taguniversaltransformer = "taguniversaltransformer"


class Evaluator(BaseTrainer):
    """docstring for Trainer."""

    def set_args(self):
        """
        get_args
        """
        # fmt: off
        super().set_args()
        parser = self.parser
        parser.add_argument('--dataset', required=True, type=Data, choices=list(Data))
        parser.add_argument('--max_seq_len', default=128, type=int)
        parser.add_argument('--max_decode_len', default=128, type=int)
        parser.add_argument('--decode_beam_size', default=5, type=int)
        parser.add_argument('--arch', required=True, type=Arch, choices=list(Arch))
        parser.add_argument('--indtag', default=False, action='store_true', help='separate tag from source string')
        parser.add_argument('--targets_included', default=False, action='store_true', help='targets included in file')
        parser.add_argument('--decode', default=Decode.greedy, type=Decode, choices=list(Decode))
        parser.add_argument('--bestacc', default=False, action='store_true', help='select model by accuracy only')
        parser.add_argument('--out_prefix', default='out', type=str)
        parser.add_argument('--src_vocab', default=None, type=str)
        parser.add_argument('--trgt_vocab', default=None, type=str)
        # fmt: on

    def load_data(self, dataset, train, dev, test):
        assert self.data is None
        logger = self.logger
        params = self.params
        # fmt: off
        if params.arch == Arch.hardmono:
            if dataset == Data.sigmorphon17task1:
                self.data = dataloader.AlignSIGMORPHON2017Task1(train, dev, test, params.shuffle)
            elif dataset == Data.g2p:
                self.data = dataloader.AlignStandardG2P(train, dev, test, params.shuffle)
            elif dataset == Data.news15:
                self.data = dataloader.AlignTransliteration(train, dev, test, params.shuffle)
            else:
                raise ValueError
        else:
            if dataset == Data.sigmorphon17task1:
                if params.indtag:
                    self.data = dataloader.TagSIGMORPHON2017Task1(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2017Task1(train, dev, test, params.shuffle)
            elif dataset == Data.unimorph:
                if params.indtag:
                    self.data = dataloader.TagUnimorph(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.Unimorph(train, dev, test, params.shuffle)
            elif dataset == Data.sigmorphon19task1:
                assert isinstance(train, list) and len(train) == 2 and params.indtag
                self.data = dataloader.TagSIGMORPHON2019Task1(train, dev, test, params.shuffle)
            elif dataset == Data.sigmorphon19task2:
                assert params.indtag
                self.data = dataloader.TagSIGMORPHON2019Task2(train, dev, test, params.shuffle)
            elif dataset == Data.g2p:
                self.data = dataloader.StandardG2P(train, dev, test, params.shuffle)
            elif dataset == Data.p2word:
                self.data = dataloader.StandardP2Word(train, dev, test, params.shuffle)
            elif dataset == Data.infp2word:
                self.data = dataloader.InferenceP2Word(train, dev, test, params.shuffle, params.src_vocab, params.trgt_vocab)
            elif dataset == Data.p2g:
                self.data = dataloader.StandardP2G(train, dev, test, params.shuffle)
            elif dataset == Data.news15:
                self.data = dataloader.Transliteration(train, dev, test, params.shuffle)
            elif dataset == Data.histnorm:
                self.data = dataloader.Histnorm(train, dev, test, params.shuffle)
            elif dataset == Data.sigmorphon16task1:
                if params.indtag:
                    self.data = dataloader.TagSIGMORPHON2016Task1(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.SIGMORPHON2016Task1(train, dev, test, params.shuffle)
            elif dataset == Data.lemma:
                if params.indtag:
                    self.data = dataloader.TagLemmatization(train, dev, test, params.shuffle)
                else:
                    self.data = dataloader.Lemmatization(train, dev, test, params.shuffle)
            elif dataset == Data.lemmanotag:
                self.data = dataloader.LemmatizationNotag(train, dev, test, params.shuffle)
            else:
                raise ValueError
        # fmt: on
        logger.info("src vocab size %d", self.data.source_vocab_size)
        logger.info("trg vocab size %d", self.data.target_vocab_size)
        logger.info("src vocab %r", self.data.source[:500])
        logger.info("trg vocab %r", self.data.target[:500])


    def load_state_dict(self, filepath):
        state_dict = torch.load(filepath)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.logger.info(f"load from {filepath}")

    def setup_evalutator(self):
        arch, dataset = self.params.arch, self.params.dataset
        if arch == Arch.hardmono:
            if dataset == Data.news15:
                self.evaluator = util.PairTranslitEvaluator()
            elif dataset == Data.sigmorphon17task1:
                self.evaluator = util.PairBasicEvaluator()
            elif dataset == Data.g2p:
                self.evaluator = util.PairG2PEvaluator()
            else:
                raise ValueError
        else:
            if dataset == Data.news15:
                self.evaluator = util.TranslitEvaluator()
            elif dataset == Data.g2p:
                self.evaluator = util.G2PEvaluator()
            elif dataset == Data.p2g:
                self.evaluator = util.P2GEvaluator()
            elif dataset == Data.histnorm:
                self.evaluator = util.HistnormEvaluator()
            else:
                self.evaluator = util.BasicEvaluator()

    def decode_target(self, write_fp, decode_fn):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance("test")
        decode_fn.reset()
        with open(f"{write_fp}.tsv", "w") as fp:
            fp.write("prediction\ttarget\tloss\tdist\n")
            for src, trg in tqdm(sampler(), total=nb_instance):
                pred, _ = decode_fn(self.model, src)
                dist = util.edit_distance(pred, trg.view(-1).tolist()[1:-1])

                src_mask = dummy_mask(src)
                trg_mask = dummy_mask(trg)
                data = (src, src_mask, trg, trg_mask)
                loss = self.model.get_loss(data).item()

                trg = self.data.decode_target(trg)[1:-1]
                pred = self.data.decode_target(pred)
                fp.write(f'{" ".join(pred)}\t{" ".join(trg)}\t{loss}\t{dist}\n')
                cnt += 1
        decode_fn.reset()
        self.logger.info(f"finished decoding {cnt} {mode} instance")

    def select_model(self):
        best_res = [m for m in self.models if m.evaluation_result][0]
        best_acc = [m for m in self.models if m.evaluation_result][0]
        best_devloss = self.models[0]
        for m in self.models:
            if not m.evaluation_result:
                continue
            if (
                type(self.evaluator) == util.BasicEvaluator
                or type(self.evaluator) == util.G2PEvaluator
                or type(self.evaluator) == util.P2GEvaluator
                or type(self.evaluator) == util.HistnormEvaluator
            ):
                # [acc, edit distance / per ]
                if (
                    m.evaluation_result[0].res >= best_res.evaluation_result[0].res
                    and m.evaluation_result[1].res <= best_res.evaluation_result[1].res
                ):
                    best_res = m
            elif type(self.evaluator) == util.TranslitEvaluator:
                if (
                    m.evaluation_result[0].res >= best_res.evaluation_result[0].res
                    and m.evaluation_result[1].res >= best_res.evaluation_result[1].res
                ):
                    best_res = m
            else:
                raise NotImplementedError
            if m.evaluation_result[0].res >= best_acc.evaluation_result[0].res:
                best_acc = m
            if m.devloss <= best_devloss.devloss:
                best_devloss = m
        if self.params.bestacc:
            best_fp = best_acc.filepath
        else:
            best_fp = best_res.filepath
        return best_fp, set([best_fp])

    def decode(self, write_fp, decode_fn=None):
        self.model.eval()
        cnt = 0
        sampler, nb_instance = self.iterate_instance("test")
        decode_fn.reset()
        with open(f"{write_fp}.tsv", "w") as fp:
            for src, trg in tqdm(sampler()):
                pred, _ = decode_fn(self.model, src)

                pred = self.data.decode_target(pred)
                fp.write(f'{"".join(pred)}\n')
                cnt += 1
        decode_fn.reset()
        self.logger.info(f"finished decoding {cnt} instance")

        
def main():
    """
    main
    """
    evaluator = Evaluator()
    params = evaluator.params
    decode_fn = get_decode_fn(
        params.decode, params.max_decode_len, params.decode_beam_size
    )
    evaluator.load_data(params.dataset, [],[], params.test)
    evaluator.setup_evalutator()
    if params.load and params.load != "0":
        if params.load == "smart":
            evaluator.smart_load_model(params.model)
        else:
            evaluator.load_model(params.load)
    else:  
        evaluator.logger.fatal("Must have trained model to decode")
    if params.targets_included:
        assert params.dataset != infp2word
        evaluator.decode_target(params.out_prefix, decode_fn)
    else:
        evaluator.decode(params.out_prefix, decode_fn)


if __name__ == "__main__":
    main()
