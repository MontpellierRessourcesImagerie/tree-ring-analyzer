BACKGROUND_RECT_WIDTH = 400
INTERPOLATION_INTERVAL = 50

Overlay.remove
run("Select None");
run("Duplicate...", " ");
run("16-bit");
run("HiLo");
width = getWidth();
height = getHeight();
makeRectangle(width-1-BACKGROUND_RECT_WIDTH, 0, BACKGROUND_RECT_WIDTH, BACKGROUND_RECT_WIDTH);
mean = getValue("Mean");
run("Select None");
run("Subtract...", "value=" + mean);
backgroundCorrectedImageID = getImageID();
run("Scale...", "x=0.064 y=0.064 width=419 height=389 interpolation=Bilinear average create");
selectImage(backgroundCorrectedImageID);
close();
run("Median...", "radius=2");
run("8-bit");
run("Auto Local Threshold", "method=Sauvola radius=15 parameter_1=0 parameter_2=0 white");
thresholdedImageID = getImageID();
run("Connected Components Labeling", "connectivity=4 type=[16 bits]");
selectImage(thresholdedImageID);
close();
connectedComponentsID = getImageID();
run("Label Size Filtering", "operation=Greater_Than size=200");
selectImage(connectedComponentsID);
close();
sizeFilterImageID = getImageID();
run("Label Morphological Filters", "operation=Dilation radius=2 from_any_label");
selectImage(sizeFilterImageID);
close();
dilatedLabelsID = getImageID();
run("Remove Border Labels", "left right top bottom");
selectImage(dilatedLabelsID);
close();
setThreshold(1, 65535);
run("Convert to Mask");
maskImageID = getImageID();
run("Skeletonize (2D/3D)");
removedBordersImageID = getImageID();
run("Analyze Skeleton (2D/3D)", "prune=none prune_0 calculate");
selectImage(removedBordersImageID);
close();
selectImage("Tagged skeleton");
close();
setThreshold(1, 255);
run("Convert to Mask");
run("Dilate");
roiManager("reset");
run("Create Selection");
roiManager("add");
roiManager("Split");
roiManager("Select", 0);
roiManager("Delete");
numberOfRois = roiManager("count");
toBeRemoved = newArray(0);
for (i = 0; i < numberOfRois; i++) {
	if (i%2 == 1) {
		roiManager("select", i);
		run("Enlarge...", "enlarge=-1 pixel");
		roiManager("update");
		continue;
	}
	toBeRemoved = Array.concat(toBeRemoved, i);
}
roiManager("select", toBeRemoved);
roiManager("delete");
run("Select None");
roiManager("Show All without labels");
run("From ROI Manager");
imageWithRoisID = getImageID();
run("Scale...", "x=- y=- width="+width+" height="+height+" interpolation=Bilinear average create");
selectImage(imageWithRoisID);
close();
Overlay.copy
close();
Overlay.paste
roiManager("reset");
run("To ROI Manager");
Overlay.remove
numberOfRois = roiManager("count");
for (i = 0; i < numberOfRois; i++) {
	roiManager("select", i);
	run("Interpolate", "interval=" + INTERPOLATION_INTERVAL + " smooth");
	roiManager("update");
}
run("From ROI Manager");
roiManager("reset");
