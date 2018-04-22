1. Change the video name in wrapper line 11 to change input video. Run the wrapper function to generate output video 
2. Depend on the type of faces present in the video, change classifier file in detectFace line 19
	For frontal face use the default ('haarcascade_frontalface_default.xml')
	For profile face use 'haarcascade_profileface.xml'
3. Depend on the brightness of the video change threshold value in  getFeatures line 31
	Default is 0.0005 
	TheMartian-0.0005
	MarquesBrownLee-0.0005
	StrangerThings-0.0005
	TyrionLannister-0.0005
4. Output video will be stored in the same folder in the name 'output.mp4'(output.mp4 will be overwritten when a new video is ran, change the output video name to store it)