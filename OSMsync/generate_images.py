import urllib
from osgeo import gdal,osr,ogr
import os
from osgeo.gdalconst import *
import numpy as np
import sys
import cv2
import skimage.io

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------

# Input files
input_geotiff_files = []
input_geotiff_files.append("../../CDIO/sydney/ortho_blue/0_1_0_tex.tif")
input_geotiff_files.append("../../CDIO/sydney/ortho_green/0_1_0_tex.tif")
input_geotiff_files.append("../../CDIO/sydney/ortho_nir/0_1_0_tex.tif")
input_geotiff_files.append("../../CDIO/sydney/ortho_pan/0_1_0_tex.tif")
input_geotiff_files.append("../../CDIO/sydney/ortho_red/0_1_0_tex.tif")
input_osm_file = "../../CDIO/sydney/ortho_blue/0_1_0_dsm.tif"
# Output files
output_geotiff_cutout_files = []
output_geotiff_cutout_files.append("vricon_ortho_blue")
output_geotiff_cutout_files.append("vricon_ortho_green")
output_geotiff_cutout_files.append("vricon_ortho_nir")
output_geotiff_cutout_files.append("vricon_ortho_pan")
output_geotiff_cutout_files.append("vricon_ortho_red")
output_osm_file = "vricon_dsm"

output_rasterized_file = "rasterized.png"
output_rasterized_cutout_file = "rasterized"

output_blended_file = "blended_whole.png"
output_blended_cutout_file = "blended"

# Test 1000x1000, from 2000,1000
# Train 2000x2000, from 1500,5500
# Train 1200x1200, from 3300,700

# Output coutout size
cutout_xmin = 3300 # Left bound, px
cutout_ymin = 700 # Top bound, px
cutout_xsize = 1000 # Vertical bound size, px
cutout_ysize = 1000 # Horizontal bound size, px

# Output file resolution
pixel_size = 0.5 #(Should be 0.5 to correspond with Vricon images)

# Download new OSM data with overpass API or use old file (1 for yes, 0 for no)
download_new_osm_data = 1

# Set which layers to draw in rasterized file (1 for yes, 0 for no)
draw_roads = 1
draw_water = 1

# ---------------------------------------------------------------
# Declare temp file lists
shape_file = []
transformed_file = []

# Temp files
osm_file = "osm_cutout.osm"
shape_file.append("shape_input_1.shp")
shape_file.append("shape_input_2.shp")
shape_file.append("shape_input_3.shp")
transformed_file.append("shape_transformed_1.shp")
transformed_file.append("shape_transformed_2.shp")
transformed_file.append("shape_transformed_3.shp")

# ---------------------------------------------------------------
# Import GeoTiff and extract GeoInfo
# ---------------------------------------------------------------
# Import GeoTiff
print '---------- Import image files ----------'
print 'Input image filename: ' + input_geotiff_files[0]
input_geotiff_data = gdal.Open(input_geotiff_files[0])

if input_geotiff_data is not None:
	# Get GeoTiff origin (bottom left) coordinates
	geotransform = input_geotiff_data.GetGeoTransform()
	vricon_origin_X = geotransform[0]
	vricon_origin_Y = geotransform[3]
	x_min = vricon_origin_X
	y_max = vricon_origin_Y
	x_max = x_min + geotransform[1]*input_geotiff_data.RasterXSize
	y_min = y_max + geotransform[5]*input_geotiff_data.RasterYSize
	print 'Input image size:' + str(int((1/pixel_size)*(x_max - x_min))) + 'x' + str(int((1/pixel_size)*(y_max - y_min))) + 'px'

	# Transform GeoTiff corner points to OSM coordinates
	vricon_srs = osr.SpatialReference()
	vricon_srs.ImportFromEPSG(32756)
	osm_srs = osr.SpatialReference()
	osm_srs.ImportFromEPSG(4326)
	ct_vricon_osm = osr.CoordinateTransformation(vricon_srs,osm_srs)

	width = input_geotiff_data.RasterXSize
	height = input_geotiff_data.RasterYSize
	gt = input_geotiff_data.GetGeoTransform()
	minx = gt[0]
	miny = gt[3] + width*gt[4] + height*gt[5]
	maxx = gt[0] + width*gt[1] + height*gt[2]
	maxy = gt[3]

	# Get the coordinates in lat, long
	latlong_min = ct_vricon_osm.TransformPoint(minx - 10000,miny - 10000)
	latlong_max = ct_vricon_osm.TransformPoint(maxx + 10000,maxy + 10000)
else:
	print 'Error! Input image file not found:' + input_geotiff_files[0]
	print '---------- Exit ----------'
	sys.exit()

# ---------------------------------------------------------------
# Extract OSM with overpass-api
# ---------------------------------------------------------------
print '---------- Extract OSM data ----------'
if download_new_osm_data == 1:
	print 'Downloading new OSM file'
	print 'OSM data bounds: ' + str(latlong_min[0]) + ',' + str(latlong_min[1]) + ',' + str(latlong_max[0]) + ',' + str(latlong_max[1])
	api_query = "http://www.overpass-api.de/api/xapi_meta?*%5Bbbox=" + str(latlong_min[0]) + "," + str(latlong_min[1]) + "," + str(latlong_max[0]) + "," + str(latlong_max[1]) + "%5D"
	osm_xml = urllib.urlopen(api_query).read()
	print 'Write OSM data to file'
	osmfile = open(osm_file, 'w')
	osmfile.write(osm_xml)
	osmfile.close()
	print 'Created OSM file:', osm_file
else:
	if os.path.exists(osm_file):
		print 'Using old OSM file:', osm_file
	else:
		print "Tried to use old OSM file, but file not found:", osm_file
		print "---------- Exit ----------"
		sys.exit()

# ---------------------------------------------------------------
# Generate shape files from osm
# ---------------------------------------------------------------
print "---------- Extract shape files ----------"
print "Extract the following layers:"
extract_sql_queries = []
# Roads:
if draw_roads == 1:
	print "   Roads"
	extract_sql_queries.append("select highway from lines where highway!='footway' and highway!='cycleway' and highway!='steps' and highway!='path' and highway!='pedestrian'")
	extract_sql_queries.append("select * from lines where highway='primary'")
	#extract_sql_queries.append("select * from lines where highway='residential'")
	#extract_sql_queries.append("select * from lines where highway='secondary'")

if draw_water == 1:
	print "   Water"
	extract_sql_queries.append("select * from multipolygons where natural='water'")

for i in range(0,len(shape_file)):
	extract_string = 'ogr2ogr -q -overwrite -f "ESRI Shapefile" ' + shape_file[i] + ' ' + osm_file + ' -progress -sql "' + extract_sql_queries[i] + '"'
	os.system(extract_string)

# ---------------------------------------------------------------
# Transform source shape files into same projection as Vricon image
# ---------------------------------------------------------------
print "---------- Transform shape files ----------"
for i in range(0,len(shape_file)):
	print 'Transform:', shape_file[i]
	source_ds = ogr.Open(shape_file[i])
	source_layer = source_ds.GetLayer()
	source_srs = source_layer.GetSpatialRef()
	source_extent = source_layer.GetExtent()
	print 'Shape file extent:', source_extent

	ct_osm_vricon = osr.CoordinateTransformation(source_srs, vricon_srs)
	source_min_xy_meters = ct_osm_vricon.TransformPoint(source_extent[0], source_extent[2])
	source_max_xy_meters = ct_osm_vricon.TransformPoint(source_extent[1], source_extent[3])
	source_extent_x_meter = int((1/pixel_size)*(source_max_xy_meters[0] - source_min_xy_meters[0]))
	source_extent_y_meter = int((1/pixel_size)*(source_max_xy_meters[1] - source_min_xy_meters[1]))
	print "Corresponding raster size in resolution " + str(pixel_size) + " meter/pixel: " + str(source_extent_x_meter) + "x" + str(source_extent_y_meter)
	transformstring = "ogr2ogr -q -t_srs EPSG:32756 -overwrite " + transformed_file[i] + " " + shape_file[i]
	os.system(transformstring)

# Get layer of transformed shape file
transformed_ds = []
transformed_layer = []

for i in range(0,len(transformed_file)):
	transformed_ds.append(ogr.Open(transformed_file[i]))
	transformed_layer.append(transformed_ds[i].GetLayer())

# ---------------------------------------------------------------
# Rasterize
# ---------------------------------------------------------------
print "---------- Rasterize shape files ----------"
# Calculate pizel size for new raster
x_res = int((x_max - x_min) / pixel_size)
y_res = int((y_max - y_min) / pixel_size)
print "Size to rasterize: " + str(x_res) + "x" + str(y_res) + "px"

print "Rasterize: " + transformed_file[0] + ", " + transformed_file[1]

# Create new raster with 3 raster bands (RGB)
driver = gdal.GetDriverByName( 'MEM' )
target_ds = driver.Create( '', x_res, y_res, 1, gdal.GDT_Byte) # For color: change 1 to 3
target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

# Create array filled with 255(white) and write into new rasterband
raster = np.zeros( (y_res, x_res), dtype=np.uint8 )
raster.fill(255)
target_ds.GetRasterBand(1).WriteArray( raster )
#target_ds.GetRasterBand(2).WriteArray( raster ) #For color
#target_ds.GetRasterBand(3).WriteArray( raster ) #For color

# ---------------------------------------------------------------
# Draw surface objects to image file with GDAL RasterizeLayer
# ---------------------------------------------------------------
gdal.RasterizeLayer(target_ds, [1], transformed_layer[2], burn_values = [120], options = ["ALL_TOUCHED=FALSE", "MERGE_ALG=ADD"]) # For color: Change to 3 bands and 3 values
# Write to file
gdal.GetDriverByName('PNG').CreateCopy(output_rasterized_file,target_ds)
target_ds = None
f = open(output_rasterized_file)
f.close

# ---------------------------------------------------------------
# Draw line objects to image file with OpenCV
# ---------------------------------------------------------------
image_data = gdal.Open(output_rasterized_file)
gt = image_data.GetGeoTransform()
img = cv2.imread(output_rasterized_file)
def build_point_list( input_shape_data ):
	input_shape_layer = input_shape_data.GetLayer(0)

	# Build list of all points in all geometries
	geometries = []
	for index in xrange(input_shape_layer.GetFeatureCount()):
		points = []
		feature = input_shape_layer.GetFeature(index)
		geometry = feature.GetGeometryRef()
		for i in range(0,geometry.GetPointCount()):
			point = geometry.GetPoint(i)
			# Calculate pixel coordinates from real coordinates
			x = (point[0]-gt[0])/gt[1]
			y = (point[1]-gt[3])/gt[5]
			points.append([int(x),int(y)])
		geometries.append(points)
	return geometries

def draw_solid_line(geometries, width, color ):
	for i in range(0, len(geometries)):
		for j in range(0, len(geometries[i])-1):
			cv2.line(img, (geometries[i][j][0],geometries[i][j][1]),(geometries[i][j+1][0],geometries[i][j+1][1]),color,width)

# Roads
road_shape_data = ogr.Open(transformed_file[0])
if road_shape_data is not None:
	road_geom = build_point_list(road_shape_data)
	draw_solid_line(road_geom, 15, (0)) # For color: Change to 3 values

road_shape_data = ogr.Open(transformed_file[1])
if road_shape_data is not None:
	road_geom = build_point_list(road_shape_data)
	draw_solid_line(road_geom, 22, (0)) # For color: Change to 3 values

# Draw Image
cv2.imwrite(output_rasterized_file, img)
road_shape_data = None
water_shape_data = None

# ---------------------------------------------------------------
# Blend GeoTiff and rasterized image
# ---------------------------------------------------------------
print "---------- Blend GeoTiff and rasterized image ----------"
weight = 0.7 # Weight for GeoTiff image, [0.0 - 1.0]
vricon_image = cv2.imread(input_geotiff_files[i])
if vricon_image.shape == img.shape:
	blend_img = cv2.addWeighted(img,1-weight,vricon_image,weight,0)
	cv2.imwrite(output_blended_file,blend_img)
else:
	print "GeoTiff and rasterized not same size. Skip blending."
# ---------------------------------------------------------------
# Crop both goal and rasterized file
# ---------------------------------------------------------------
print "---------- Write cutout files ----------"
if (cutout_xmin + cutout_xsize > x_res) or (cutout_ymin + cutout_ysize > y_res):
	print "Error! Cutout out of image bounds"
	print '---------- Exit ----------'
	sys.exit()

osm = skimage.io.imread(input_osm_file, plugin='tifffile')
x_range = int(np.floor(x_res/cutout_xsize)*cutout_xsize)
y_range = int(np.floor(y_res/cutout_ysize)*cutout_ysize)
counter = 1
for k in range(0,x_range, int(cutout_xsize)):
	cutout_xmin = k
	for j in range(0,y_range, int(cutout_ysize)):
		cutout_ymin = j
		c = str(counter)
		print "Cutout size: " + str(cutout_xsize) + " x " + str(cutout_ysize) + " px"
		print 'Cut image 1 of 7'
		cropstring = "gdal_translate -q -srcwin " + str(cutout_xmin) + " " + str(cutout_ymin) + " " + str(cutout_xsize) + " " + str(cutout_ysize) + " " + output_rasterized_file + " " + output_rasterized_cutout_file + c + ".png"
		os.system(cropstring)

		print 'Cut image 2 of 7'
		cropstring = "gdal_translate -q -srcwin " + str(cutout_xmin) + " " + str(cutout_ymin) + " " + str(cutout_xsize) + " " + str(cutout_ysize) + " " + output_blended_file + " " + output_blended_cutout_file + c + ".png"
		os.system(cropstring)

		for i in range(0,len(output_geotiff_cutout_files)):
			print 'Cut image ' + str(i+2) + ' of ' + str(len(output_geotiff_cutout_files) + 2)
			cropstring = "gdal_translate -q -srcwin " + str(cutout_xmin) + " " + str(cutout_ymin) + " " + str(cutout_xsize) + " " + str(cutout_ysize) + " " + input_geotiff_files[i] + " " + output_geotiff_cutout_files[i] + c + ".png"
			os.system(cropstring)

		''' save height map '''
		cutout = osm[k:k+cutout_xsize,j:j+cutout_ysize]
		cutout = cv2.convertScaleAbs(cutout)
		cv2.imwrite(str(output_osm_file + str(counter) + '.tif'), cutout)
		counter = counter +1
		print counter

# ---------------------------------------------------------------
# Close datasets and clean temp files
# ---------------------------------------------------------------
source_ds = None
for i in range(0,len(transformed_ds)):
	transformed_ds[i] = None
	os.system("del " + shape_file[i][0:-4] + "*")
	os.system("del " + transformed_file[i][0:-4] + "*")

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
print '---------- Done ----------'
