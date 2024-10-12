# Talking_Face_Dataset_Preprocess_script
In the realm of video processing, accurately and swiftly locating and cropping the speaking person's face from a video is a common yet challenging task. This technical blog post details my journey in optimizing a script that utilizes FFmpeg for face detection and cropping, focusing on enhancing its processing speed and output quality, with a special emphasis on pre-processing datasets for speaking face recognition.

### Initial Code Analysis

We started with a solid foundation provided by the original face cropping code (fomm), which laid a great base. However, despite its strengths, the original code had several issues that needed addressing:

### Issues and Solutions

#### 1. Slow Processing Speed

**Issue:** The original code processes the video frame by frame, which is extremely slow.

**Solution:** To expedite the process, I introduced a strategy of frame skipping. Specifically, the code now detects faces every 10 frames (a variable that can be adjusted). This approach significantly reduces processing time. However, this also introduces a challenge: skipping frames might lead to missing critical frame changes, such as transitions from a face to no face. To mitigate this, I implemented a backtracking mechanism. Once a change in face presence is detected, the code retraces and processes frames individually to accurately pinpoint the start and end positions of the speaking face.

#### 2. High Computational Cost for High-Resolution Videos

**Issue:** The original method consumes a lot of computing power, especially for 1080p videos.

**Solution:** To reduce the computational load, I resized each frame to 480p during the reading process before detection and then restored the FFmpeg parameters to the original size afterward. This not only减轻了计算负担 but also maintained sufficient resolution for accurate face detection.

#### 3. Inadequate Output Parameters

**Issue:** The original method's output parameters were not optimal. The output video was named "crop.mp4," which is inflexible and does not account for multiple speakers in a single video.

**Solution:** I improved the generation of output parameters. Now, the output video's name will be based on the input video's name, with an additional "crop" suffix and an index to differentiate between multiple speaking faces in a video. For example, if the input video is named "inputvideo.mp4," the output will be named "inputvideo_1_crop.mp4," "inputvideo_2_crop.mp4," and so on.

### Conclusion

Through these optimizations, we not only improved the speed of video processing but also enhanced the quality and flexibility of the output video. I hope these enhancements will be helpful to you when dealing with video face detection and cropping tasks. If you have any questions or want to discuss further, please feel free to leave a comment below.

Taking a 1080P video as an example, the processing speed can reach 300 FPS (A 40), which is about 100 times faster than the original Fomm speed of approximately 3 FPS.



# **优化视频人脸定位与裁剪：提升性能与输出质量**

在视频处理领域，准确快速地定位并裁剪出视频中的说话人脸是一项常见但挑战性的任务。本文将分享我如何优化一个基于ffmpeg的人脸裁剪脚本，以提高其处理速度和输出质量。

### 原始代码分析

我们从一段优秀的基础代码开始，该代码旨在定位视频中的说话人脸并返回相应的ffmpeg参数。然而，尽管基础扎实，原始代码在处理速度和输出参数方面存在一些不足。

### 问题与解决方案

#### 1. 处理速度慢

**问题：** 原始代码对视频进行逐帧处理，这在处理高分辨率视频时尤其耗时。

**解决方案：** 为了提高处理速度，我引入了跳帧处理的策略。具体来说，代码现在会每处理10帧检测一次人脸（这个数字可以根据需要调整）。这种方法显著减少了处理时间，但同时也带来了一个新的挑战：跳帧可能会导致在人脸出现和消失的瞬间错过关键帧。为了解决这个问题，我添加了一个回溯机制，一旦检测到人脸状态的变化，代码会回溯并逐帧处理，以确保准确捕捉到说话人脸的起始和终止位置。

#### 2. 高分辨率视频处理

**问题：** 对于1080p等高分辨率视频，原始代码的计算消耗非常大。

**解决方案：** 为了减少计算量，我在读取每一帧时先将其尺寸调整为480p，然后再进行人脸检测。这样既减少了计算负担，又保持了足够的分辨率以进行准确的人脸检测。在检测完成后，我会将ffmpeg的参数调整回原始尺寸，以确保输出视频的质量。

#### 3. 输出参数优化

**问题：** 原始代码的输出参数存在一些问题，比如输出视频的命名不够灵活，且没有考虑到一个视频中可能存在多个说话人脸的情况。

**解决方案：** 我改进了输出参数的生成方式。现在，输出视频的名称将基于输入视频的名称，并加上“crop”后缀，同时附加一个索引，以区分视频中的多个说话人脸。例如，如果输入视频名为“inputvideo.mp4”，输出视频将被命名为“inputvideo_1_crop.mp4”，“inputvideo_2_crop.mp4”等。

### 结语

通过这些优化，我们不仅提高了视频处理的速度，还提升了输出视频的质量和灵活性。希望这些改进对你处理视频人脸定位与裁剪任务时有所帮助。

### 性能分析

以1080P的视频为例，处理速度可达到300 FPS（A 40），相比较原始的Fomm速度约3FPS，提高了约100倍。
