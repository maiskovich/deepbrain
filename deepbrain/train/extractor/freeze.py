import tensorflow as tf
import sys


def run(model_dir, name):
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.compat.v1.Session()
        ckpt_path = tf.train.latest_checkpoint(model_dir)
        saver = tf.compat.v1.train.import_meta_graph('{}.meta'.format(ckpt_path))
        saver.restore(sess, ckpt_path)

        # Pick your own nodes
        output_node_names = ["img", "training", "prob", "pred", "dim"]

        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, tf.compat.v1.get_default_graph().as_graph_def(), output_node_names)

        output_graph = "{}.pb".format(name)
        with tf.io.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            sess.close()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
