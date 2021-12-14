.. _mzutils: https://pypi.org/project/mzutils/

`mzutils`_ is a personal toolkit that contains various methods to do miscellaneous jobs related to data cleaning, preprocessing, visualizing, evaluating and modeling in various fields of machine learning, especially NLP and RL. It supports Datasets like SQuAD, GLUE, Deepmind CNN/DailyMail in NLP and d4rl in offlineRL, as well as their conversions to JSON, CSV, numpy arrays and tensors.

The purpose of this toolkit is to reuse the code and make Mohan Zhang's life easier.

detailed documentation can be found `here <mzutils.rtfd.io>`_.

Install
-------

- if you want to use only the minimal functionalities, please do

.. code-block::

    $ pip install -U mzutils

- or if you want to use nlp specific functions, please do
  
.. code-block::
    
    $ pip install -U mzutils[nlp]

- or if you want to use all the functionalities, please do
  
.. code-block::
    
    $ pip install -U mzutils[all]
    $ pip install -r security_requirements.txt

.. note::
    due to this `unreasonable feature of pip <https://github.com/pypa/pip/issues/6301>`_, you will need to check and install from `requirements.txt <https://github.com/Mohan-Zhang-u/mzutils/blob/master/requirements.txt>`_, `extras_requirements.txt <https://github.com/Mohan-Zhang-u/mzutils/blob/master/extras_requirements.txt>`_ or `security_requirements.txt <https://github.com/Mohan-Zhang-u/mzutils/blob/master/security_requirements.txt>`_ according to your specific need.
