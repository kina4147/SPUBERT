In compliance with the obligation to adhere to the Apache 2.0 license, we hereby provide the following notice:

* The Apache 2.0 license is available at the site below.
  - https://www.apache.org/licenses/LICENSE-2.0
  - https://spdx.org/licenses/Apache-2.0.html
  - https://opensource.org/licenses/Apache-2.0

*  Unless required by applicable law or agreed to in writing, software distributed under the Apache 2.0 license is distributed on an "as is" basis, without warranties or conditions of any kind, either express or implied. 
   Please see the License for what this License permits or restricts.


* The following modifications have been made to the provided software:
  - **model/modeling_spubert.py**: We modified the tensor dimensions and module sizes of "modeling_bert.py" in "transformers/src/transformers/models/bert" of the huggingface to handle multi-pedestrian trajectory and semantic map utilized by SPUBERT. 
  - **model/spubert.py**: We modified "configuration_bert.py" and "modeling_bert.py" in "transformers/src/transformers/models/bert" of huggingface. Specifically, we added configuration parameters related to the trajectory prediction, and separated configuration classes for TGP and MGP modules. We modified the inputs, outputs, network architecture, losses, etc. of the main modules in "modeling_bert.py" of the huggingface to handle multi-pedestrian trajectory and semantic map. In addition, we newly made SPUBERT by connecting two BERT models and adding CVAE in between.


* The followings pertain to copyrights, patents, trademarks, and attribution of the software provided:
  - **model/modeling_spubert.py**: 
    - Copyright (c) 2023 Electronics and Telecommunications Research Institute (ETRI)
    - Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
    - Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
  - **model/spubert.py**: 
    - Copyright (c) 2023 Electronics and Telecommunications Research Institute (ETRI)
    - Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
    - Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.


* Contact information
  - Name: Ki-In Na
  - Affiliation: Field Robotics Research Section, Electronics and Telecommunications Research Institute (ETRI) 
  - E-mail: kina4147@etri.re.kr
  - Phone: +82-42-860-3929
