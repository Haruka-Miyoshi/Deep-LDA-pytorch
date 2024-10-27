# **Deep-LDA**
LDA (Latent Dirichlet Allocation) is a statistical machine learning model designed for document classification, commonly known as a topic model. It operates by identifying themes or 'topics' within a collection of documents, allowing for the categorization of text based on shared underlying themes. Through this process, LDA enables the extraction of meaningful insights from large text corpora, aiding in tasks such as information retrieval, document clustering, and recommendation systems.<br>
In recent years, VAE (Variational Autoencoder)-based models have also been applied to topic modeling. Unlike traditional LDA, which relies on a probabilistic graphical model framework, VAE-based models leverage neural networks to learn the latent topic distribution of documents. This approach enables more flexible and expressive topic modeling, making it effective for large-scale data and complex, non-linear data structures. Additionally, using VAE allows for document embeddings and can be applied to text generation, further broadening its potential applications.[srivastava2017](https://arxiv.org/abs/1703.01488).<br>
This repository provides a program implementing Deep LDA in PyTorch.<br>
In `main.ipynb`, the repository also includes training on the MNIST dataset using the DeepMLDA module, along with evaluations of reconstructed images and assessments of the latent variable space.<br>
The repository [is0383kk](https://github.com/is0383kk/Dirichlet-VAE) was used as a reference for the implementation. I extend my sincere gratitude for this resource.<br>

## **Reference**
```
@misc{srivastava2017autoencodingvariationalinferencetopic,
      title={Autoencoding Variational Inference For Topic Models}, 
      author={Akash Srivastava and Charles Sutton},
      year={2017},
      eprint={1703.01488},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1703.01488}, 
}
```