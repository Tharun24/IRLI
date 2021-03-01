# Iterative Re-partitioning for Learning to Index

It learns to partition and map together using a single neural network via an alternative training and re-partitioning.

IRLI index creation takes 3 steps. First(left)(initialization step)- the labels are pooled randomly into B buckets using a 2-universal hash function. The figure shows only five buckets (while we have a few thousands in practice). Second(middle)- We train R fully-connected networks on N data points, where any bucket containing at-least one true labels is positive. Third(right): After training for a few epochs, the labels are re-assigned to the buckets. For each label, we provide a representative input to the R networks. We selectthe top-K buckets and assign the label to the least occupied bucket (K=2 in the figure yields 2nd and 3rd buckets as the top-scored ones. Light-green bucket is the lesser occupied one, and hence we assign the label to the 3rd bucket). A larger K ensures perfect load balance, while a smaller K ensures higher precision and recall.
<p>
<img src="/NeuralIndex.png" style="width:640px;height:480px;" align="center">
</p>

IRLI query process. Here the query vector is passed through R trained  models, and  each  one  gives  the  probability scores over the corresponding buckets. Figure shows m= 1 for illustration purpose. The top candidates are sorted based on the aggregated scores of each label.
<p>
<img src="/NeaurlIndexQuery.png" style="width:640px;height:480px;" align="center">
</p>

Prerequisites:
You are expected to have TensorFlow 1.x installed (1.8 - 1.14 should work) and have atleast 2 GPUs with 32GB memory (or 4 GPUs with 16 GB memory). We will add support for TensorFlow 2.x in subsequent versions. Cython is also required for importing a C++ function gather_batch during evaluation (if you cannot use C++ for any reason, please refer to the Cython vs Python for evaluation section below). sklearn is required for importing murmurhash3_32 (from sklearn.utils). Although the version requirements for cython and sklearn are non that stringent as Tensorflow, use Cython-0.29.14 and sklearn-0.22.2 in case you run into any issues.

1) Tensorflow- 1.8 - 1.14 should work
2) Minimum 1 GPU, preferred >4 GPUs with 32GB memory
3) Memory required - Around 200 GB

Instructions to run:
1) Edit the config.py for required parameters. 
2) Run preprocess.py
3) Run train_script.py
4) For inference, run approx_eval.py



