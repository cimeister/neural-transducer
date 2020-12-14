# Phoneme to Word Neural Transducer

This is a branch of Shijie Wu's [`neural-transducer` library](https://github.com/shijie-wu/neural-transducer.git) 


## Setup

- Environment (conda): `environment.yml`
- Pre-commit check: `pre-commit run --all-files`
- Compile: `make`


## Run

If using pretrained model checkpoints, create a `checkpoints` directory and unzip binaries, e.g.,

```
mkdir checkpoints
unzip models.zip -r checkpoints/
```

After compiling, you can run an example of phoneme to word (graphemes) using the following command

```
bash p2word_decode.sh
```

which uses phonemes in `cmudict/input_example.tsv` as an example. New models can be trained using the trm-p2word.sh script in `examples/transformer` where training data can be pulled/processed using the script in the `cmudict` directory. Directory names in the p2word_decode script may need to be changed accordingly.

