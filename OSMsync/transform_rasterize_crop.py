import urllib
from osgeo import gdal,osr,ogr
import os
from osgeo.gdalconst import *
import numpy as np
import sys

# Input files
goal_file = "0_0_0_tex.tif"
source_file_roads = "roads_exp_large.shp"
source_file_parks = "natural_exp_large.shp"

# Output files
goal_cropped_file = "geotiff_cropped.tif"
cropped_file = "sydney_cropped.tiff"

# Temp files
transformed_file_roads = "sydney_transformed_roads.shp"
transformed_file_parks = "sydney_transformed_parks.shp"
rasterized_file = "sydney.tiff"

# Settings ( =0.5 to correspond with Vricon images)
pixel_size = 0.5

# Transform source shape, roads, file into same projection as Vricon image
print 'Transform:', source_file_roads
source_ds_roads = ogr.Open(source_file_roads)
source_layer_roads = source_ds_roads.GetLayer()
source_srs = source_layer_roads.GetSpatialRef()
source_extent = source_layer_roads.GetExtent()
print 'Shape file extent:', source_extent
vricon_sr = osr.SpatialReference()
vricon_sr.ImportFromEPSG(32756)
ct_osm_vricon = osr.CoordinateTransformation(source_srs, vricon_sr)
source_min_xy_meters = ct_osm_vricon.TransformPoint(source_extent[0], source_extent[2])
source_max_xy_meters = ct_osm_vricon.TransformPoint(source_extent[1], source_extent[3])
source_extent_x_meter = int((1/pixel_size)*(source_max_xy_meters[0] - source_min_xy_meters[0]))
source_extent_y_meter = int((1/pixel_size)*(source_max_xy_meters[1] - source_min_xy_meters[1]))
print 'Corresponding raster size in resolution', pixel_size, 'meter/pixel:',source_extent_x_meter , 'x', source_extent_y_meter
transformstring = "ogr2ogr -t_srs EPSG:32756 -overwrite " + transformed_file_roads + " " + source_file_roads
os.system(transformstring)

# Transform source shape, parks, file into same projection as Vricon image
print 'Transform:', source_file_parks
source_ds_parks = ogr.Open(source_file_roads)
source_layer_parks = source_ds_parks.GetLayer()
source_srs = source_layer_parks.GetSpatialRef()
source_extent = source_layer_roads.GetExtent()
print 'Shape file extent:', source_extent
vricon_sr = osr.SpatialReference()
vricon_sr.ImportFromEPSG(32756)
ct_osm_vricon = osr.CoordinateTransformation(source_srs, vricon_sr)
source_min_xy_meters = ct_osm_vricon.TransformPoint(source_extent[0], source_extent[2])
source_max_xy_meters = ct_osm_vricon.TransformPoint(source_extent[1], source_extent[3])
source_extent_x_meter = int((1/pixel_size)*(source_max_xy_meters[0] - source_min_xy_meters[0]))
source_extent_y_meter = int((1/pixel_size)*(source_max_xy_meters[1] - source_min_xy_meters[1]))
print 'Corresponding raster size in resolution', pixel_size, 'meter/pixel:',source_extent_x_meter , 'x', source_extent_y_meter
transformstring = "ogr2ogr -t_srs EPSG:32756 -overwrite " + transformed_file_parks + " " + source_file_parks
os.system(transformstring)

# Import GeoTiff
print 'Import:', goal_file
vricon_data = gdal.Open(goal_file)

# Get GeoTiff origin (bottom left) coordinates 
geotransform = vricon_data.GetGeoTransform()
vricon_origin_X = geotransform[0]
vricon_origin_Y = geotransform[3]
x_min = vricon_origin_X
y_max = vricon_origin_Y
x_max = x_min + geotransform[1]*vricon_data.RasterXSize
y_min = y_max + geotransform[5]*vricon_data.RasterYSize
print 'Input GeoTiff size:', int((1/pixel_size)*(x_max - x_min)), 'x', int((1/pixel_size)*(y_max - y_min))

# Get extent of transformed shape file
transformed_ds_roads = ogr.Open(transformed_file_roads)
transformed_layer_roads = transformed_ds_roads.GetLayer()

# Get extent of transformed shape file
transformed_ds_parks = ogr.Open(transformed_file_parks)
transformed_layer_parks = transformed_ds_parks.GetLayer()

# Calculate pizel size for new raster
x_res = int((x_max - x_min) / pixel_size)
y_res = int((y_max - y_min) / pixel_size)
print 'Size to rasterize:', x_res, 'x', y_res

# Create a new rasterband whith desired resolution and size
print 'Rasterize:', transformed_file_roads, ',', transformed_file_parks
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
gdal.RasterizeLayer(target_ds, [1,3], transformed_layer_parks, burn_values = [1,1], options = ["ALL_TOUCHED=TRUE", "MERGE_ALG=ADD"])
gdal.RasterizeLayer(target_ds, [2,3], transformed_layer_roads, burn_values = [1,1], options = ["ALL_TOUCHED=TRUE", "MERGE_ALG=ADD"])
gdal.RasterizeLayer(target_ds, [1], transformed_layer_roads, burn_values = [255], options = ["ALL_TOUCHED=TRUE", "MERGE_ALG=ADD"])
gdal.GetDriverByName('GTiff').CreateCopy(rasterized_file,target_ds)

# Approx Vricon Center Coordinates (vriconbild ca 4000*4000 meter)
vricon_center_X = vricon_origin_X + 2000
vricon_center_Y = vricon_origin_Y - 2000

# Approx Vricon Corner Coordinates
vricon_top = vricon_center_Y + 2500
vricon_bottom = vricon_center_Y - 1000
vricon_left = vricon_center_X - 2000
vricon_right = vricon_center_X + 500

# Crop both goal and rasterized file
warpstring = "gdalwarp -overwrite -te " + str(vricon_left) + " " + str(vricon_bottom) + " " + str(vricon_right) + " " + str(vricon_top) + " " + goal_file + " " + goal_cropped_file
os.system(warpstring)
warpstring = "gdalwarp -overwrite -te " + str(vricon_left) + " " + str(vricon_bottom) + " " + str(vricon_right) + " " + str(vricon_top) + " " + rasterized_file + " " + cropped_file
os.system(warpstring)

# Close datasets
transformed_ds_roads = None
transformed_ds_parks = None

# Clean temp files
os.system("del " + transformed_file_roads)
os.system("del " + transformed_file_parks)

print "Done"
