






**Multimodal Sequence Modelling for Visual Story Prediction Using Attention-Based Fusion**

**Module Leader:** Alejandro Jimenez Rodriguez

**Module Name:** Deep Neural Networks and Learning Systems

**Module Code:** 55-710365


# Table of Contents
[1.	Introduction	1](#_toc229504821)

[2.	Dataset Description	1](#_toc229504822)

[3.	Exploratory Data Analysis	2](#_toc229504823)

[4.	Data preprocessing	6](#_toc229504824)

[5.	Baseline Architecture	9](#_toc229504825)

[6.	Proposed Innovation	17](#_toc229504826)

[7.	Results and Evaluation	21](#_toc229504827)

[8.	Explainability Analysis	24](#_toc229504828)

[9.	Discussion	26](#_toc229504829)

[10.	Conclusion	26](#_toc229504830)

[References	27](#_toc229504831)

[Appendix	29](#_toc229504832)




2

1. # <a name="_toc229504821"></a>**Introduction**
The field of multimodal deep learning, which involves modelling and interpreting multiple data modalities like image and text, has emerged as a crucial research direction in the artificial intelligence field (Bayoudh et al., 2021). Visual story reasoning is a difficult task that involves understanding temporal relationships, contextual information, and sequential patterns to predict meaningful story continuations. This study concentrates on creating the next element in a multimodal text-to-image narrative by leveraging image-text pairs from the StoryReasoning dataset. Guo et al. (2025) proposed a multimodal framework that combines CNN, LSTM, attention mechanisms, and fusion techniques to enhance sequence understanding and prediction accuracy. The purpose of the proposed system is to improve the performance for multimodal reasoning, temporal learning, and coherent story generation. 
1. # <a name="_toc229504822"></a>**Dataset Description**
The StoryReasoning dataset for this study was downloaded from the Hugging Face Dataset Repository. A dataset of sequential image-text pairs has been created for multimodal reasoning and visual story generation tasks. It has 4,178 stories generated from 52,016 movie frames and their associated textual descriptions and chain-of-thought annotations. The dataset is divided into training and testing sets containing 3,552 and 626 samples, respectively, in Figure 1. Each sample contains images, story descriptions, frame counts, and reasoning annotations, which can be used for multimodal sequence modelling and story prediction tasks. This dataset is licensed for research and academic use under the Creative Commons BY-ND-4.0 license.

**Dataset Link:** 

<https://huggingface.co/datasets/daniel3303/StoryReasoning> 

**Implementation Tools**

The study was implemented using Python, PyTorch, Torchvision, HuggingFace Datasets, NumPy, Matplotlib, and PIL. Preprocessing, multimodal learning, training, and visualization tasks were performed in a Google Colab environment with YAML-based configuration management.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.001.png)

*Figure *1*: Preview of the Dataset.*
1. # <a name="_toc229504823"></a>**Exploratory Data Analysis**
The distribution of frame counts by story sequences is shown in Figure 2. The number of frames in most stories is around 16 to 18 frames, and in a few stories, it is almost 5 to 10 frames available in the dataset.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.002.png)

*Figure *2*: Frame Count Distribution.*

The distribution of story text length by word count is shown in Figure 3. The descriptions of most stories are in the range of 500 to 1500 words, though a few sequences have more than 2000 words in the dataset.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.003.png)

*Figure *3*: Story Length Distribution.*

Movie frames are displayed sequentially for one storyline in the samples of Figure 4. The movement from one frame to the other shows that this is a temporal continuity extending across the sequence of the picture's visibilities, where the contextual changes are evident. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.004.png)

*Figure *4*: Sample Sequential Frames.*

As can be seen in Figure 5, there are variations in image dimensions among the first 50 stories. The majority of images are around 240 pixels tall, with widths ranging from about 420 to 585 pixels. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.005.png)

*Figure *5*: Image Size Variation Using Scatter Plot.*

The most common words in story descriptions are shown in Figure 7, with the top 20 words appearing most frequently. The frequency of common narrative terms like “the,” “a,” and “to” are significantly higher throughout textual sequences.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.006.png)

*Figure *6*: Frequent Word Analysis.*

The mean of the percentages of vocabulary covered for various maximum text lengths is displayed in Figure 8. Based on the analysis, a higher value for the text length is chosen, 128, because of the improved coverage. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.007.png)

*Figure *7*: Vocabulary Coverage Analysis.*

A comparison of the training and testing frame count distributions is shown in Figure 9. These patterns are similar in both datasets, with greater concentrations occurring at frame sequence lengths of 5 and 17.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.008.png)

*Figure *8*: Train-Test Distribution Comparison.*

The number of characters and objects in each story is shown in Figure 10. Most stories use about 15-35 characters and 5-25 objects, and some sequences have more.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.009.png)

*Figure *9*: Character and Object Distribution.*
1. # <a name="_toc229504824"></a>**Data preprocessing** 
Based on exploratory data analysis results, the maximum text length parameter is updated to 512 tokens as shown in Figure 10. Xu et al. (2020) pointed out that configuration boosts vocabulary coverage and facilitates efficient sequence processing of text during model training. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.010.png)

*Figure *10*: Maximum Text Length Configuration.*

For image preprocessing, resizing, and conversion to tensors, as well as normalization based on mean and standard deviation values, are performed prior to multimodal model training and feature extraction processes (Asuai et al., 2025). 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.011.png)

*Figure *11*: Image Transformation Pipeline.*

The vocabulary generation process from training stories is demonstrated in Figure 12. Constructed vocabulary consists of 9,196 tokens, including all padding, unknown, start-of-sequence, and end-of-sequence tokens.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.012.png)

*Figure *12*: Vocabulary Construction Process.*

In the text encoding process shown in Figure 13, textual sequences are encoded into fixed-length tensors (Liu & Qiu, 2025). After padding and applying sequence length constraints, the output of the encoded text has 514 tokens.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.013.png)

*Figure *13*: Text Encoding Process.*

A custom StoryDataset class is created to process sequential images, target images, textual tensors, and frame count to support multimodal sequence modelling tasks as illustrated in Figure 14.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.014.png)

*Figure *14*: StoryDataset Class Defining.*

The splitting of the dataset shown in Figure 15 separates the subset into three sets, training, validation, and testing with 400, 50, and 50 samples respectively, for model evaluation.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.015.png)

*Figure *15*: Dataset Splitting Process.*
1. # <a name="_toc229504825"></a>**Baseline Architecture**
A pretrained ResNet18 backbone is employed in the visual encoder architecture shown in Figure 16 to extract visual features from sequential image frames and produce feature representations of dimension size 256 (Tur et al., 2023).

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.016.png)

*Figure *16*: Visual Encoder Architecture.*

The Text Encoder described in Figure 17 uses embedding and LSTM layers to extract textual features (Naik & Jaidhar, 2022). Generated output representation has output shape values of (2, 256), which include hidden dimension features. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.017.png)

*Figure *17*: Text Encoder Architecture.*

The baseline fusion module in Figure 18 is a simple concatenation and linear projection fusion module, which produces fused multimodal features of shape (2, 5, 256). 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.018.png)

*Figure *18*: Baseline Fusion Module.*

The sequence modelling component shown in Figure 19 is a temporal learning-based approach using LSTM across multimodal sequences. The sequence output shape is (2, 5, 256), and the dimension of the hidden states is 256. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.019.png)

*Figure *19*: Sequence Modelling Architecture.*

The attention mechanism illustrated in Figure 20 produces contextual feature representations of shape (2, 256) and attention weights of shape (2, 5). The total sum of the calculated attention weights is 1.0000. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.020.png)

*Figure *20*: Attention Mechanism Architecture.*

The architecture of the image decoder shown in Figure 21 employs the transposed convolution layers for image reconstruction (Boucherit & Kheddar, 2025). The output shape of the image prediction generated after upsample operations is (2, 3, 224, 224). 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.021.png)

*Figure *21*: Architecture of Image Decoder.*

The text decoder in Figure 22 produces text predictions via embedding and LSTM layers (Gu et al., 2024). The shape of the output tensor is (2, 514, 9196), which corresponds to the vocabulary-level token prediction probabilities.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.022.png)

*Figure *22*: Text Decoder Architecture.*

The baseline model in Figure 23 combines visual and text encoding, fusion, sequence modelling, attention, image decoding, and text decoding modules, having 23,301,920 trainable parameters. The generated image prediction shape is (2, 3, 224, 224), while the text prediction shape is (2, 514, 9196). 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.023.png)

*Figure *23*: Complete Baseline Architecture.*
1. # <a name="_toc229504826"></a>**Proposed Innovation**
The cross-modal attention fusion architecture shown in Figure 24 allows the use of visual and textual features to interact with each other through bidirectional attention operations (Chen et al., 2025). After feature integration, the output shape of the generated fusion is (2, 5, 256).

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.024.png)

*Figure *24*: Cross-Modal Attention Fusion.*

The Bidirectional LSTM sequence model shown in Figure 29 is a model that processes temporal sequences both forwards and backward (Yang et al., 2025). The output shape of the generated sequence is (2, 5, 256), and the hidden dimension size is set to 256.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.025.png)

*Figure *25*: Bidirectional Sequence Model.*

The multi-task loss mechanism shown in Figure 30 leverages learnable weighting parameters to balance the image and text prediction losses (Sharma et al., 2025). The computed total loss is 11.6341, with the image loss being 2.0092 and the text loss being 9.6249. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.026.png)

*Figure *26*: Multi-Task Loss Function.*

The entire innovation model shown in Figure 31 contains 23,500,320 trainable parameters: cross-modal attention fusion, Bidirectional LSTM modelling, attention mechanisms, and decoding components. The generated image prediction shape is (2, 3, 224, 224), while the text prediction shape is (2, 514, 9196). 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.027.png)

*Figure *27*: Complete Innovation Architecture.*
1. # <a name="_toc229504827"></a>**Results and Evaluation**
The baseline model training and validation loss curves are shown in Figure 28. The curves both show a consistent decline over time, suggesting stable learning patterns and gradual improvement across epochs in the multimodal model training. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.028.png)

*Figure *28*: Baseline Loss Curves.*

The innovation model loss curves in Figure 34 revealed the consistent decrease in both training and validation losses as training was progressed, revealing better convergence and a stable optimization performance across epochs.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.029.png)

*Figure *29*: Innovation Model Loss Curves.*

The performance of the baseline model and the innovation model for various evaluation metrics is presented in Figure 35. The innovation model obtains a lower loss of 1.1775, a lower perplexity of 1.4326, and a higher BLEU-4 score of 0.9238. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.030.png)

*Figure *30*: Evaluation Metrics Comparison.*

The performance metric visualizations displayed in Figure 36 are used to provide a comparison between baseline and innovation model outputs. The BLEU-4 score is improved to 0.9238, perplexity is decreased to 1.4326, and the reconstruction quality metrics are comparable to the others.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.031.png)

*Figure *31*: Performance Metrics Visualization.*

The training and validation performance differences between baseline and innovation models are shown in the comparative loss curves in Figure 37. The innovation model has lower loss values and better convergence behaviour. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.032.png)

*Figure *32*: Loss Curve Comparison.*

The model improvement analysis in Figure 38 indicates the improvement in performance of the innovation model. It enhances loss by 20.2%, perplexity by 22.3%, and the BLEU-4 score by 7.8%.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.033.png)

*Figure *33*: Model Improvement Analysis.*
1. # <a name="_toc229504828"></a>**Explainability Analysis**
The attention weight visualization in Figure 39 brings attention to high importance in the frame level of the multimodal sequence. Frame 1 has the maximum attention value of 0.1098, while Frame 9 has the minimum attention value of 0.0549. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.034.png)

*Figure *34*: Attention Weight Visualization.*

The attention distribution analysis in Figure 40 compares the attention weights of different story sequences that include 11, 16, and 18 frames. The initial frames usually have higher attention values in multimodal reasoning processes.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.035.png)

*Figure *35*: Attention Distribution Analysis.*

The attention level heatmap in Figure 41 is a visualization of the intensity of the attention to each frame in multiple stories. Darker areas show higher attention values, and lighter areas show lower importance of the context in sequence modelling tasks. 

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.036.png)

*Figure *36*: Attention Weight Heatmap.*

A cross-modal fusion activation heatmap shown in Figure 42 displays the sequences of feature activation across 16 frames. The activation regions are larger for stronger multimodal interactions between visual and textual feature representations.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.037.png)

*Figure *37*: Cross-Modal Activation Heatmap.*
1. # <a name="_toc229504829"></a>**Discussion**
The results obtained show that the proposed innovation model had better performance on the problem of learning a sequence in multimodal learning than the baseline architecture. Cross-modal attention fusion led to a better combination of visual and textual features, and the bidirectional LSTM enhanced temporal sequence understanding. The innovation model had lower validation loss and a higher BLEU-4 score on evaluation. The same attention visualizations additionally revealed that the model effectively captured attention to key frames in story sequences, which led to enhanced interpretability and contextual reasoning for multimodal predictions.
1. # ` `**<a name="_toc229504830"></a>Conclusion**
The study proposed a multimodal deep learning model for story reasoning based on sequential images and text descriptions from StoryReasoning. The innovation model proposed in this paper was designed to enhance the performance of multimodal sequence understanding and prediction by fusing cross-modal attention, Bidirectional LSTM, and attention mechanisms. Experimental evaluation has shown lower losses, lower perplexity, and higher BLEU-4 scores than the baseline architecture. Model interpretability was also enhanced through attention visualization and heatmap analysis, which were used to highlight some of the most important frames and feature interactions in multimodal reasoning tasks. 




# <a name="_toc229504831"></a>**References**
Asuai, C., Andrew, M., Prince, A. A., Ogheneochuko, D. E., Joseph-Brown, Aghoghovia Agajere, Merit, I., & Collins, A. (2025). Transformer-Augmented Deep Learning Ensemble for Multi-Modal Neuroimaging-Based Diagnosis of Amyotrophic Lateral Sclerosis - FTS Digilib. *Futuretechsci.org*. <https://dl.futuretechsci.org/id/eprint/134/1/14661-Article%20Text-51782-1-10-20251013.pdf> 

Bayoudh, K., Knani, R., Hamdaoui, F., & Mtibaa, A. (2021). A survey on deep multimodal learning for computer vision: advances, trends, applications, and datasets. *The Visual Computer*. <https://doi.org/10.1007/s00371-021-02166-7> 

Chen, K., Lin, X., Xia, T., & Bai, R. (2025). Research on Park Perception and Understanding Methods Based on Multimodal Text–Image Data and Bidirectional Attention Mechanism. *Buildings*, *15*(9), 1552. <https://doi.org/10.3390/buildings15091552> 

Gu, W. jun, Zhong, Y. hao, Li, S. zun, Wei, C. song, Dong, L. ting, Wang, Z. yue, & Yan, C. (2024). *Predicting Stock Prices with FinBERT-LSTM: Integrating News Sentiment Analysis*. 67–72. <https://doi.org/10.1145/3694860.3694870> 

Guo, W., Liu, S., Weng, L., & Liang, X. (2025). Power Grid Load Forecasting Using a CNN-LSTM Network Based on a Multi-Modal Attention Mechanism. *Applied Sciences*, *15*(5), 2435–2435. <https://doi.org/10.3390/app15052435> 

Liu, F., & Qiu, H. (2025). *Context Cascade Compression: Exploring the Upper Limits of Text Compression*. ArXiv.org. <https://arxiv.org/abs/2511.15244> 

Naik, D., & Jaidhar, C. D. (2022). A novel Multi-Layer Attention Framework for visual description prediction using bidirectional LSTM. *Journal of Big Data*, *9*(1). <https://doi.org/10.1186/s40537-022-00664-6> 

Sharma, R., Chen, C., Chang, F.-J., Yun, S., Xie, X., Meng, R., Xu, D., Mottini, A., & Cui, Q. (2025). Multi-Modal Multi-Task Unified Embedding Model (M3T-UEM): A Task-Adaptive Representation Learning Framework. *Thecvf.com*, 22783–22793. <https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_Multi-Modal_Multi-Task_Unified_Embedding_Model_M3T-UEM_A_Task-Adaptive_Representation_Learning_ICCV_2025_paper.html> 

Tur, A. O., Dall’Asen, N., Beyan, C., & Ricci, E. (2023). Unsupervised Video Anomaly Detection with Diffusion Models Conditioned on Compact Motion Representations. *Lecture Notes in Computer Science*, 49–62. <https://doi.org/10.1007/978-3-031-43153-1_5> 

Xu, L., Zhang, X., & Dong, Q. (2020). *CLUECorpus2020: A Large-scale Chinese Corpus for Pre-training Language Model*. ArXiv.org. <https://arxiv.org/abs/2003.01355> 

Yang, T., Cheng, Y., Ren, Y., Lou, Y., Wei, M., & Xin, H. (2025). A Deep Learning Framework for Sequence Mining with Bidirectional LSTM and Multi-Scale Attention. *Proceedings of the 2025 2nd International Conference on Innovation Management and Information System*, 472–476. <https://doi.org/10.1145/3745676.3745751> 

Boucherit, I., & Kheddar, H. (2025). Reinforced Residual Encoder–Decoder Network for Image Denoising via Deeper Encoding and Balanced Skip Connections. *Big Data and Cognitive Computing*, *9*(4), 82. <https://doi.org/10.3390/bdcc9040082> 


# <a name="_toc229504832"></a>**Appendix** 
Additional preprocessing outputs, baseline training logs, innovation model training logs, and cleaned text examples are included in the appendix to provide supporting implementation details and supplementary experimental information.

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.038.png)

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.039.png)

**

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.040.png)

![](Aspose.Words.66c0f813-c920-4e65-801b-2ce1c9f7d9e6.041.png)

