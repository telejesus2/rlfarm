import torch.nn as nn
from typing import List

from rlfarm.networks.utils import tie_weights, init_weights
from rlfarm.networks.blocks import Conv2DBlock, DenseBlock, ConvTranspose2DBlock, IdBlock, log_block
from rlfarm.utils.logger import Summary

AVAILABLE_NETWORKS = [
    'cnn',
    'dense',
    'dcnn',
    'id',
    'custom',
]


class Network(nn.Module):     
    def forward(self, x, detach=False):
        pass

    def copy_weights_from(self, source):
        pass

    def log(self, prefix) -> List[Summary]:
        pass


class BlockNetwork(Network):
    def __init__(self, blocks: List[nn.Module]):
        super(BlockNetwork, self).__init__()
        self.blocks = nn.ModuleList(blocks)

        # weight initialization 
        for block in self.blocks: init_weights(block, block.activation)
        
    def forward(self, x, detach=False):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i==0 and detach: # detach first block
                x = x.detach()          
        return x

    def copy_weights_from(self, source, block_index=0):
        """Tie layers of a block"""
        assert block_index < len(self.blocks)

        for i in range(len(self.blocks[block_index].layers)):
            tie_weights(src=source.blocks[block_index].layers[i],
                        trg=self.blocks[block_index].layers[i])

    def log(self, prefix) -> List[Summary]:
        summaries = []
        for _, block in enumerate(self.blocks):
            summaries += log_block(block, prefix)
        return summaries


def make_cnnetwork(
    input_dim, output_dim,
    filters=[40,40], kernels=[3,3], strides=[2,2], paddings=[2,2], dilations=[1,1],
    conv_norm=None, conv_act='relu', conv_drop_rate=0.0,
    out_norm=None, out_act='relu', out_drop_rate=0.0,
):
    """Convolution layers + flatten operator + one fully connected layer    
    """

    conv_block = Conv2DBlock(
        input_shape=input_dim, tag='conv',
        filters=filters, kernels=kernels, strides=strides, paddings=paddings, dilations=dilations,
        norm=conv_norm, act=conv_act, drop_rate=conv_drop_rate
    )
    out_block = DenseBlock(
        input_dim=conv_block.output_dim, output_dim=output_dim, tag='out',
        hidden_nodes=[],
        norm=out_norm, act=out_act, drop_rate=out_drop_rate,
    )

    return BlockNetwork([conv_block, out_block])


def make_dense_network(
    input_dim, output_dim,
    hidden_nodes=[64,64],
    fc_norm=None, fc_act='relu', fc_drop_rate=0.0,
    out_norm=None, out_act='relu', out_drop_rate=0.0,
):
    """Fully connected layers + one fully connected layer    
    """

    dense_block = DenseBlock(
        input_dim=input_dim, output_dim=None, tag='fc',
        hidden_nodes=hidden_nodes,
        norm=fc_norm, act=fc_act, drop_rate=fc_drop_rate,
    )
    out_block = DenseBlock(
        input_dim=dense_block.output_dim, output_dim=output_dim, tag='out',
        hidden_nodes=[],
        norm=out_norm, act=out_act, drop_rate=out_drop_rate,
    )

    return BlockNetwork([dense_block, out_block])


def make_dcnnetwork(
    input_dim, output_dim,
    filters=[40,40], kernels=[3,3], strides=[2,2], paddings=[2,2], dilations=[1,1],
    conv_norm=None, conv_act='relu', conv_drop_rate=0.0,
    in_norm=None, in_act='relu', in_drop_rate=0.0,
):
    """One fully connected layer + unflatten operator + ConvTranspose layers
    """

    conv_block = ConvTranspose2DBlock(
        output_shape=output_dim, tag='deconv',
        filters=filters, kernels=kernels, strides=strides, paddings=paddings, dilations=dilations,
        norm=conv_norm, act=conv_act, drop_rate=conv_drop_rate
    )
    in_block = DenseBlock(
        input_dim=input_dim, output_dim=conv_block.input_dim, tag='in',
        hidden_nodes=[],
        norm=in_norm, act=in_act, drop_rate=in_drop_rate,
    )

    return BlockNetwork([in_block, conv_block])


def make_idnetwork():
    return BlockNetwork([IdBlock(tag='id')])


def make_custom_network(input_dim, output_dim, module: str, object: dict):
    import importlib.util
    print(module)
    spec = importlib.util.spec_from_file_location(
        name='custom',  # name is not related to the file, it's the module name!
        location=module  # full path to the script
    )
    mymodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mymodule)
    class_ = getattr(mymodule, object['class'])
    return class_(**object['kwargs'])


def make_network(class_, kwargs):
    assert class_ in AVAILABLE_NETWORKS
    
    if class_ == 'cnn':
        return make_cnnetwork(**kwargs)
    elif class_ == 'dense':
        return make_dense_network(**kwargs)
    elif class_ == 'dcnn':
        return make_dcnnetwork(**kwargs)
    elif class_ == 'id':
        return make_idnetwork()
    elif class_ == 'custom':
        return make_custom_network(**kwargs)