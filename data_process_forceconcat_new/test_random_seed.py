import tensorflow as tf

rng = tf.random.Generator.from_seed(123, alg='philox')


def ns():
    seed = rng.make_seeds(2)[0]

    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # print(new_seed)
    return new_seed


for i in range(5):
    print(ns())

# tf.random.stateless_normal(shape=[3], seed=new_seeds[0, :])
