import json

import cv2
import numpy as np
import onnxruntime as ort

from .rois import DeviceAutoRois


def main():
    sess = ort.InferenceSession("./arcaea_ml/arcaea_jackets.onnx", providers=["CPUExecutionProvider"])  # type: ignore  # noqa: E501
    data = json.loads(sess.get_modelmeta().custom_metadata_map["relation"])
    idx2id, id2idx = data["idx2id"], data["id2idx"]

    rois = DeviceAutoRois(cv2.imread("./arcaea_ml/images/screen3.png"))
    rescaled_jacket = cv2.resize(rois.jacket, (60, 60))
    jacket = np.moveaxis(rescaled_jacket, -1, 0)

    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: [jacket.astype(np.float32)]})[0].flatten()
    classId = np.argmax(out)
    print(idx2id[classId])
    cv2.imshow("hmm", rescaled_jacket)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
