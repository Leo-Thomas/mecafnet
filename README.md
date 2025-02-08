# Multi-encoder ConvNeXt Network with Attentional Feature Fusion for Multispectral Semantic Segmentation

<!-- ABOUT THE PROJECT -->
## About the project

This work presents an encoder-decoder architecture designed for Land Cover Classification utilizing Multispectral Imagery. The architecture consists of two parallel feature extraction branches based on ConvNeXt, one dedicated to processing RGB information and the other to NIR data. The encoders are connected with a spectral fusion and pyramidal decoding process enhanced with a convolutional block attention module, which emphasizes the most relevant features to produce accurate segmentation masks. Experiments are conducted using the Potsdam dataset, and results demonstrate that the proposed approach achieves notable performance, with an overall accuracy of 90.83% and a mean intersection over union of 77.69%. Furthermore, visual tests show that the proposed approach is better at differentiating complex classes, achieving segmentations closer to the ground-truth. The method outperforms other well-known architectures, including U-Net, PSPNet, DeepLabV3, and DeepLabV3+, as well as other state-of-the-art approaches.

### Paper

Accepted at IEEE SoutheastCon 2025

### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Installation

pip install git+https://github.com/Leo-Thomas/mecafnet.git

## Data set

We used the ISPRS Potsdam dataset which can be found for free [here](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITAITON -->
## Citation

```
Soon

```
<!-- CONTACT -->
## Contact

Leo Thomas Ramos - [LinkedIn](https://www.linkedin.com/in/leo-thomas-ramos/) - leo.ramos@kauel.com

Angel Sappa - [LinkedIn](https://es.linkedin.com/in/angel-sappa-61532b17) - asappa@cvc.uab.cat


<p align="right">(<a href="#top">back to top</a>)</p>
