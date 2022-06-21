""" import re


string = "tesstt_mask_1.png"
string2 = "mask_"
m = re.search(r"mask_[0-9]", string2)


if m:
    print(m.group(0))
 """

from model.UNET import UNET, block_conv_down
import tensorflow as tf


""" example = tf.random.uniform(shape=(572, 572, 1))
example = tf.expand_dims(example, axis=0)
print(example.shape)
example, p = block_conv_down(example, 64, maxpool=True)
print(example.shape)
print(p.shape)
 """
model = UNET((256, 256, 1))

sample = tf.random.uniform(shape=(256, 256, 1))
sample = tf.expand_dims(sample, axis=0)
out = model(sample)
print(out.shape)
