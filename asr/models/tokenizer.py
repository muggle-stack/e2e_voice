import onnxruntime as ort
import numpy as np

class Tokenizer:
    def __init__(
            self,
            ortext_path,
            decode_model_path
        ):
        self.so = ort.SessionOptions()
        self.so.register_custom_ops_library(ortext_path)
        self.so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(decode_model_path, self.so)

    def decode(self,
               input_ids):
        decoder_inputs = dict(
            ids=np.array(
                input_ids,
                dtype=np.int64),
            fairseq=np.array([False], dtype=np.bool_))
        return self.sess.run(None, decoder_inputs)