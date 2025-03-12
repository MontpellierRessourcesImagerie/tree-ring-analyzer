THRESHOLD = 0.0840;
SIZE = 900;
CLOSING = 64;

run("Duplicate...", " ");
setOption("BlackBackground", true);
//setThreshold(THRESHOLD, 1e+30);
setAutoThreshold("Otsu dark no-reset");
run("Convert to Mask");
maskID = getImageID();
run("Connected Components Labeling", "connectivity=4 type=[16 bits]");
connectedComponentsID = getImageID();
run("Label Size Filtering", "operation=Greater_Than size="+SIZE);
setThreshold(1, 65535);
run("Convert to Mask");
filteredMaskID = getImageID();
run("Morphological Filters", "operation=Closing element=Octagon radius="+CLOSING);
run("Skeletonize (2D/3D)");

//selectImage(maskID);
//close();

//selectImage(connectedComponentsID);
//close();

//selectImage(filteredMaskID);
//close();
