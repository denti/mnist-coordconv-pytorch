# coding: utf-8
import onnxruntime as rt
import numpy
from PIL import Image


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = numpy.exp(numpy.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


def main():

    img = Image.open('../data/2.pgm')
    
    x = numpy.array(img)
    x = ((1-(x/255.0))- 0.1307) / 0.3081
    x = numpy.repeat(numpy.expand_dims(numpy.expand_dims(x, 0), 0), 6000, axis=0)

    sess = rt.InferenceSession("../models/mnist_cc.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    pred = sess.run([output_name], {input_name: x.astype(numpy.float32)})[0]
    
    print(list(map(lambda x: "%.4f" % x, softmax(pred[0]))))

if __name__ == "__main__":
    main()
