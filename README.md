### Fun with frequencies
![Alt text]https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FHybrid_image&psig=AOvVaw0L-iPWAl5lcefHc3x-hxmf&ust=1708079415772000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCOjpiO6RrYQDFQAAAAAdAAAAABAJ
## Table of Content
  * [Description](#Description)
  * [Motivation](#Motivation)
  * [Features](#Features)
  * [Getting Started](#Getting-Started)
  * [Installation](#installation)
  * [Directory Tree](#Directory-Tree)
  * [Contributing](#Contributing) 
  * [Acknowledgments Contributing](#Acknowledgments-Contributing)
  * [License](#license)
  
## Description
This repository contains my advanced image preprocessing code, which includes techniques for creating **hybrid images**, constructing **Gaussian and Laplacian stacks and pyramid with and without downsampling**, and executing **multiresolution blending**. These methods are crucial for a wide range of applications including **image enhancement, editing, and computational photography**.My goal is to provide a comprehensive set of scripts and modules that enable researchers, developers, and image processing enthusiasts to tackle complex preprocessing tasks with ease.

## Motivation
This project was born out of my **_freelance work_** in **image processing** and a desire to create a comprehensive solution that addresses the complex needs of **image enhancement and manipulation**. Throughout my freelancing journey, I encountered numerous challenges that required advanced image preprocessing techniques. Realizing the lack of accessible well structured code that integrate these advanced features, I was motivated to write it, that not only serves my clients' needs but also contributes to the broader community. By sharing my work, I hope to empower others to achieve high-quality image processing results, inspire further innovation in the field, and foster a collaborative environment where ideas and techniques can be exchanged freely.

## Features
This  includes the following key features:

1. **_Hybrid Images:_** I offer a method to create images that change interpretation based on the viewing distance.
2. **_Multiresolution Blending:_** I implement a technique to seamlessly blend images together, maintaining the integrity of details from each source image through pyramid representations.
3. **_Gaussian and Laplacian Stacks and pyramids:_** I provide code  to manipulate image details efficiently across different scales Automation in saving results.
4. **_Report Analysis HTML_:** code for display results



### Getting Started
## Prerequisites
You will need the following to use this labs:

1. Python 3.8 or newer
2. NumPy
3. OpenCV
4. SciPy (for advanced mathematical functions)
5. PIL 
6. os
7. skimage 
8. matplotlib



## Installation
To get started with this code repository, follow these steps:

The Code is written in Python 3.12.1 If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
1. Clone the repository:
```bash
git clone https://github.com/Rizwanali324/fun_with_frequency.git
```
2. Navigate to the project directory:
 ```bash 
 cd fun_with_frequency
  ```

3. Install rrequirementse:
 ```bash
  pip install -r requirements.txt
   ```

 ## Directory Tree 
```
├── code 
│   ├── align_images.py
│   ├── crop_image.py
│   ├── hybrid_image_starter.py
│   ├── hybrid_image.py
│   ├── main_accentuation.py
│   ├── main_melange.py
│   ├── main_pile.py
│   └── stacks.py
├── web
│   ├──images
│   ├──results
│             ├──accentuation
│             ├──hybrid
│             ├──melange
│             ├──pile
├──enmly.html
├──hybrid.html
├──index.html
├──linlconl.html
├──water.html
├── requirements.txt
├── LICENSE
├── README.md
```



### Contributing
Contributions to this repository are more than welcome. If you have ideas for improvement or have identified bugs, please follow this standard process:

1. Fork the project
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a pull request
### License
This project is licensed under the MIT License - see the LICENSE.md file for details.

### Acknowledgments Contributing
I extend my gratitude to all who have inspired, guided, and supported my journey in image processing, contributing to the development of this code repository.

