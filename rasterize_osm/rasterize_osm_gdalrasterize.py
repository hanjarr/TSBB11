import urllib
from osgeo import gdal,osr,ogr
import os
from osgeo.gdalconst import *
import numpy as np
import sys

# Declare file lists
shape_file = []
transformed_file = []

# Input files
# MUST BE SPECIFIED: Filename for input GeoTiff file
input_geotiff_file = "0_0_0_tex.tif"

# Output files
output_rasterized_file = "rasterized.tiff"
output_geotiff_cutout_file = "geotiff_cutout.tif"
output_rasterized_cutout_file = "rasterized_cutout.tiff"

# Temp files
osm_file = "osm_cutout.osm"
shape_file.append("shape_input_1.shp")
shape_file.append("shape_input_2.shp")
transformed_file.append("shape_transformed_1.shp")
transformed_file.append("shape_transformed_2.shp")

# Settings 
pixel_size = 0.5 #(Should be 0.5 to correspond with Vricon images)

# Import GeoTiff
print 'Import:', input_geotiff_file
input_geotiff_data = gdal.Open(input_geotiff_file)

# Get GeoTiff origin (bottom left) coordinates 
geotransform = input_geotiff_data.GetGeoTransform()
vricon_origin_X = geotransform[0]
vricon_origin_Y = geotransform[3]
x_min = vricon_origin_X
y_max = vricon_origin_Y
x_max = x_min + geotransform[1]*input_geotiff_data.RasterXSize
y_min = y_max + geotransform[5]*input_geotiff_data.RasterYSize
print 'Input GeoTiff size:', int((1/pixel_size)*(x_max - x_min)), 'x', int((1/pixel_size)*(y_max - y_min))

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
latlong_min = ct_vricon_osm.TransformPoint(minx,miny)
latlong_max = ct_vricon_osm.TransformPoint(maxx,maxy)

# Extract OSM with overpass-api
#print 'Extract OSM data whithin coordinates:', latlong_min[0], ',', latlong_min[1], ',', latlong_max[0], ',', latlong_max[1]
#api_query = "http://www.overpass-api.de/api/xapi_meta?*%5Bbbox=" + str(latlong_min[0]) + "," + str(latlong_min[1]) + "," + str(latlong_max[0]) + "," + str(latlong_max[1]) + "%5D"
#osm_xml = urllib.urlopen(api_query).read()
#print 'Write OSM data to file'
#osmfile = open(osm_file, 'w')
#osmfile.write(osm_xml)
#osmfile.close()
#print 'Created osm file:', osm_file

# Generate shape files from osm
extract_sql_queries = []
extract_sql_queries.append("select osm_id,highway from lines where highway is not null") # Roads
#extract_sql_queries.append('select osm_id,osm_way_id,building from multipolygons where building is not null') # Buildings
extract_sql_queries.append("select * from multipolygons where natural='water'")

for i in range(0,len(shape_file)):
	extract_string = 'ogr2ogr -overwrite -f "ESRI Shapefile" ' + shape_file[i] + ' ' + osm_file + ' -progress -sql "' + extract_sql_queries[i] + '"'
	print extract_string
	os.system(extract_string)
	print ''

# Transform source shape files into same projection as Vricon image
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
	print 'Corresponding raster size in resolution', pixel_size, 'meter/pixel:',source_extent_x_meter , 'x', source_extent_y_meter
	transformstring = "ogr2ogr -t_srs EPSG:32756 -overwrite " + transformed_file[i] + " " + shape_file[i]
	os.system(transformstring)

# Get layer of transformed shape file
transformed_ds = []
transformed_layer = []

for i in range(0,len(transformed_file)):
	transformed_ds.append(ogr.Open(transformed_file[i]))
	transformed_layer.append(transformed_ds[i].GetLayer())
	
# Calculate pizel size for new raster
x_res = int((x_max - x_min) / pixel_size)
y_res = int((y_max - y_min) / pixel_size)
print 'Size to rasterize:', x_res, 'x', y_res

# Create a new rasterband whith desired resolution and size
print 'Rasterize:', transformed_file[0], ',', transformed_file[1]
driver = gdal.GetDriverByName( 'MEM' )
target_ds = driver.Create( '', x_res, y_res, 3, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))

# Create array filled with 255(white) and write into new rasterband
raster = np.zeros( (y_res, x_res), dtype=np.uint8 )
raster.fill(255)
target_ds.GetRasterBand(1).WriteArray( raster )
target_ds.GetRasterBand(2).WriteArray( raster )
target_ds.GetRasterBand(3).WriteArray( raster )

# Rasterize shapefile and append into new rasterband
gdal.RasterizeLayer(target_ds, [1,3], transformed_layer[1], burn_values = [1,1], options = ["ALL_TOUCHED=TRUE", "MERGE_ALG=ADD"])
gdal.RasterizeLayer(target_ds, [2,3], transformed_layer[0], burn_values = [1,1], options = ["ALL_TOUCHED=TRUE", "MERGE_ALG=ADD"])
gdal.RasterizeLayer(target_ds, [1], transformed_layer[0], burn_values = [255], options = ["ALL_TOUCHED=TRUE", "MERGE_ALG=ADD"])
gdal.GetDriverByName('GTiff').CreateCopy(output_rasterized_file,target_ds)

# Approx Vricon Center Coordinates (vriconbild ca 4000*4000 meter)
vricon_center_X = vricon_origin_X + 2000
vricon_center_Y = vricon_origin_Y - 2000

# Approx Vricon Corner Coordinates
vricon_top = vricon_center_Y + 1000
vricon_bottom = vricon_center_Y - 1000
vricon_left = vricon_center_X - 1000
vricon_right = vricon_center_X + 1000

# Crop both goal and rasterized file
warpstring = "gdalwarp -overwrite -te " + str(vricon_left) + " " + str(vricon_bottom) + " " + str(vricon_right) + " " + str(vricon_top) + " " + input_geotiff_file + " " + output_geotiff_cutout_file
os.system(warpstring)
warpstring = "gdalwarp -overwrite -te " + str(vricon_left) + " " + str(vricon_bottom) + " " + str(vricon_right) + " " + str(vricon_top) + " " + output_rasterized_file + " " + output_rasterized_cutout_file
os.system(warpstring)

# Close datasets and clean temp files
source_ds = None

for i in range(0,len(transformed_ds)):
	transformed_ds[i] = None
	os.system("del " + shape_file[i][0:-4] + "*")
	os.system("del " + transformed_file[i][0:-4] + "*")

print 'Done'
