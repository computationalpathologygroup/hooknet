from argconfigparser import parse
from source.model import HookNet
from source.generator.batchgenerator import RandomBatchGenerator
from source.trainer import Trainer


def train():
    config = parse('./parameters.yml')

    hooknet = HookNet(input_shape=config['input_shape'],
                      n_classes=config['n_classes'],
                      hook_indexes=config['hook_indexes'])

    batchgenerator = RandomBatchGenerator(batch_size=config['batch_size'],
                                          input_shape=hooknet.input_shape,
                                          output_shape=hooknet.output_shape,
                                          n_classes=config['n_classes'])

    trainer = Trainer(epochs=config['epochs'],
                      steps=config['steps'],
                      batch_size=config['batch_size'],
                      model=hooknet,
                      batch_generator=batchgenerator,
                      seed=config['seed'])

    trainer.train()


if __name__ == "__main__":
    train()
