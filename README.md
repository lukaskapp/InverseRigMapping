Inverse Rig Mapping
===============

1. Introduction
===============
Inverse Rig Mapping is a toolset for Maya to map skeletal animation data e.g. mocap data back to arbitrary character rigs using machine learning. It is based on the paper "Learning an inverse rig mapping for character animation". The machine learning model learns the correlation between the rig control parameters and the dependent joint parameters. After learning, it can predict these rig control parameters based on the skeletal animation data.

The common workflow for mapping skeletal animation back to rigs is to use retargeting tools like Maya's HumanIK, constraint setups, or creating custom scripts that work for the specific character rig. All of these workflows have some sort of limitation, are often time consuming and not flexible for different rigs. The Inverse Rig Mapping (IRM) tool aims to solve this problem by providing a flexible and easy-to-use workflow that works with any rig - at least as long as the model can understand the correlation between rig control and joint parameters. Currently the tool only works with simpler rigs - see the limitations section below for more details.



2. Installation
===============
The toolset has been developed and tested on Windows 10/11 using Autodesk Maya 2022.4, 2023.3 and 2024.0.1 and Python 3.10.11.


Place the Inverse Rig Mapping folder in a location of your choice and run the "install_venv.bat" file. This will create a new virtual Python environment in the IRM folder (make sure you run it in that folder), install all the necessary Python libraries and set an environment variable that is essential for the Maya installation. Make sure that a similar Python version (3.10.11 and above) is installed and that "python.exe" is registered as "python" command.

The machine learning part uses its own Python environment - different from the one Maya uses because it needs libraries that aren't available in Maya. You can use your own Python environment if you like, but make sure you have all the libraries listed in the requirements.txt file installed.


You can use your NVIDIA GPU to improve performance when using this tool. This uses the CUDA Toolkit, which needs to be installed first. You can download the latest one here: https://developer.nvidia.com/cuda-downloads

The IRM tool will dynamically use CUDA/GPU if it is available on your workstation. Otherwise it will switch to CPU - you can also force CPU in the train settings.

More information on using CUDA with PyTorch can be found here - especially if you have an older version of CUDA or a different OS: https://pytorch.org/get-started/locally/


Last but not least, install the tool in Maya: open the file "install_maya.py" in the Maya script editor and run it. This should create a shelf tool called "IRM" in the currently visible shelf. When you click on this button, the tool window will appear and you are ready to use the tool.




3. Workflow & Settings
===============
3.1 Data Generation
===============
3.1.1 Train Parameters
===============
To get started, you need data that will later be used to train the machine learning model. There are two separate areas in the interface for adding the control rig and the joint parameters. Simply select any control in your rig (NURBS curves/surfaces and meshes are accepted) and press the 'Add' button. This will add every keyable and scalar attribute of the selected control to the tree view. You can remove specific attributes by right-clicking on them and selecting 'Delete' - this also works with multi-selection. You can reset the Parameter UI using the 'Clear All' button.

The same applies to the joint parameters at the bottom. Simply select all the joints in your rig and press the 'Add' button to include every joint attribute - only joints are accepted. Any attribute that has incoming connections and is therefore influenced by other nodes will be added to the tree view. You can also delete certain joint attributes to exclude them from training. Your UI should now look something like this:

The order of the parameters doesn't matter, as they will be reordered when the training data is generated to ensure the same order every time, regardless of the selection.


3.1.2 Min/Max Range
===============
For a better prediction of the animation later on, it is important to set correct and plausible ranges for each rig parameter. Only this range will be used for training and therefore the model can only predict correct values for this range later on - so if your maximum range for translateX is 50, but your expected predicted value would be 100, the model will have a hard time predicting this as it hasn't been trained for this range. You can change the range of a single or multiple parameters by selecting them and using the "Minimum" and "Maximum" fields on the right hand side of the "Generator Settings". For rotation, you don't need to go beyond 180, as a range of -180 to 180 is the full possible range for rotations.

IMPORTANT: In general, only the given parameters are used for the prediction part, so if you exclude e.g. translateX, it won't be predicted later. The same goes for the attribute ranges you set - if the skeletal animation later goes beyond that range, it will have a hard time predicting the rig control values. Getting this right is vital for the best and most efficient result.


3.1.3 Number of Poses
===============
The number of poses defines the number of random steps between the min/max range, so 1000 means there will be 1000 different steps between the min and max value. In general the number of poses should be high enough to get a sufficient training result, more poses and therefore more data can greatly improve the training but will also increase the training time and the amount of memory needed which may be too much for the current workstation. 1000 poses is a good starting point and you can slowly go up to 5000 if you like - it always depends on the rig and the amount of parameters.


3.1.4 Output Files
===============
The last step is to specify a path where the generated data will be stored. By default this will be "IRM_folder/training_data/jnt" and "IRM_folder/training_data/jnt" - feel free to use these default paths.


3.1.5 Generate Train Data
===============
Once everything is set up, all you need to do is press the 'Generate Train Data' button. This will take a moment, depending on the amount of train parameters and the number of poses. In general, it should only take a few seconds to a few minutes. When it is finished, you will find the generated rig and joint data as CSV files in the directories provided.




3.2 Model Training
===============
3.2.1 Learning Rate
===============
The learning rate describes how fast your model is learning, essentially describing the steps through the data. If it is too high, the model may converge too quickly, potentially missing important patterns in the data. If it's too low, the model may take too long to learn and/or get stuck. Typical learning rates are between 0.1 and 0.0001.


3.2.2 Epochs
===============
This is the number of times the entire dataset is run through the model during training. If you use a small number, your model may not learn everything it needs to know (underfitting). If you use a large number, your model may start to memorise the data (overfitting), and it won't be good at making predictions about data it hasn't seen before. The optimal number of epochs is problem-specific, but a typical range is between 10 and 1000.


3.2.3 Force CPU
===============
This option forces the model to use the CPU instead of the GPU for computation. This can be useful if you are experiencing GPU limitations or don't want to use GPU acceleration.


3.2.3 Python Exe
===============
This is the path to the Python executable on your machine. The model will use this Python environment to run the script. Make sure you have all the necessary libraries installed - see above under "Installation" for more details.


3.2.4 Output Settings
===============
Here you need to specify the control rig and the joint CSV file containing the previously generated data. You will also need to specify the path where the trained model will be saved as a PyTorch (PT) file, so that you can make predictions later without having to train the model again.


3.2.5 Model Training
===============
Once you have adjusted the settings and defined the paths, you are ready to train the model by pressing the 'Train Model' button.

This is the process of feeding your data to the model and allowing it to learn from it. During training, the model attempts to minimise its prediction error (loss) through an iterative process of adjusting its internal parameters. The model adjusts these parameters based on the learning rate and the number of epochs.

During training, the goal is to minimise this loss value. This means adjusting the model parameters (training data, learning rate, epochs) so that the difference between the predicted and actual values is as small as possible. A decrease in the loss value over epochs usually indicates that the model is learning and improving its predictions. However, if the loss stops decreasing or increases, it may indicate problems such as overfitting or that the model has reached its capacity for this data.

You can find the current loss and epoch in the progress bar window that pops up during training.




3.3 Prediction / Inference
===============
3.3.1 Animation Parameters
===============
Similarly to the train data, you first need to specify all the joints (only joints are accepted) of the animated skeleton that will be mapped back to the control rig. To do this, select all the animated joints and add them using the 'Add' button. Again, check the animation parameters listed - all attributes with keys will be added automatically. You can delete them again by right-clicking or using the 'Clear All' button.

These parameters must be the same as those used for the joint training data - this means that every joint used for the training data must also be added here, otherwise the prediction will not work!

The names of the animated joints should be close to the names of the trained joints - this is important to get the same order of parameters as in the training data.


3.3.2 Train Data and Trained Model
===============
These are the file paths for the training data and the trained model respectively. The training data is the control rig and joint dataset that was used to train the model. The trained model is the result of the training process, a file that contains all the weights and biases that the model has learned. This file is used to load the model for prediction. You have specified these paths in the training process.


3.3.3 Python Exe
===============
This is the path to the Python executable on your machine. The model will use this Python environment to run the script. Make sure you have all the necessary libraries installed - see above under "Installation" for more details.


3.3.4 Mapping Prediciton
===============
Once everything is set up, you can start mapping the animation of the animated skeleton back to the rig by pressing the 'Map Prediction' button.

This is a multi-step process. First it will collect the animation data from the joints provided and modify it to work with the model. Then it will use the trained model to predict the animation values of the control rig and finally it will apply these predicted values to the rig. In the end, the animation of the control rig should closely match that of the animated joints. If not, you may need to change some settings in the training process to ensure a better prediction result.

Note: The mapping of values to the control is limited to the namespace provided in the training process. So if your control had the name "arm_L_wrist_ctrl" in the training process, then this control must have the same name in the prediction process.

Depending on the complexity of the rig, the predicted result can be quite far from the animated skeleton, or even static. This is a known limitation and bug.

As a workaround, you can split the training into different parts to reduce the complexity of the data - so only learn and map the leg, arm or spine one at a time.




4. Limitations and Bugs
===============
- The predicted rig animation can be quite different from the skeletal one - changing the number of poses, learning rate or epochs can improve the mapping.
- A large number of poses and/or many control rig and joint parameters require a lot of memory, sometimes too much for the workstation.
- Mapping skeletal data back to rigs only works with the rig control names that were used in training - so it won't work if you want to map it to rigs with different namespaces to those used in training.
- Training complex character rigs with many parameters can result in a static mapping - (almost) the same values for every frame.




5. References
===============
1. Daniel Holden, Jun Saito, Taku Komura. 2015. "Learning an inverse rig mapping for character animation" in SCA '15: Proceedings of the 14th ACM SIGGRAPH / Eurographics Symposium on Computer Animation, August 2015, Pages 165–173, https://dl.acm.org/doi/10.1145/2786784.2786788
2. Carl Edward Rasmussen, Christopher K. I. Williams. 2005. "Gaussian Processes for Machine Learning", Retrieved June 21, 2023, from https://gaussianprocess.org/gpml/chapters/RW.pdf
3. GPyTorch. 2023. "GPyTorch Documentation". Retrieved June 21, 2023, from https://docs.gpytorch.ai/en/latest/
4. PyTorch. 2023. "PyTorch Documentation Release 2.0". Retrieved June 21, 2023, from https://pytorch.org/docs/2.0/
5. Unreal Engine. 2023. "How to use the Machine Learning Deformer". Retrieved June 21, 2023, from https://docs.unrealengine.com/5.2/en-US/how-to-use-the-machine-learning-deformer-in-unreal-engine/
6. Dmitry Kostiaev. 2020. "Better rotation representations for accurate pose estimation". Retrieved June 21, 2023, from https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
7. Eric Perim, Wessel Bruinsma, and Will Tebbutt. 2021. "Gaussian Processes: from one to many outputs". Retrieved June 21, 2023, from https://invenia.github.io/blog/2021/02/19/OILMM-pt1/#:~:text=Then%2C%20a%20multi%2Doutput%20Gaussian,on%20an%20extended%20input%20space

