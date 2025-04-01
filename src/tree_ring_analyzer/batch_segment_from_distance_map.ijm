/*
 * Macro template to process multiple images in a folder
 */

THRESHOLD = 0.0840;
SIZE = 1600;
CLOSING = 64;


input = getDir("Input directory");
output = getDir("Output directory");
suffix =".tif"

// See also Process_Folder.py for a version of this code
// in the Python scripting language.

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	print("Processing: " + input + File.separator + file);
	print("Saving to: " + output);
	open(input + file);
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
	save(output + file);
	close("*");
}
