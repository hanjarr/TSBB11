from osgeo import osr, gdal
import cv2
import numpy as np
from func_cut import cut_out

# get the existing coordinate system
ds = gdal.Open('0_0_0_tex.tif')
old_cs= osr.SpatialReference()
old_cs.ImportFromWkt(ds.GetProjectionRef())

# create the new coordinate system
wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs .ImportFromWkt(wgs84_wkt)

# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs,new_cs)
inverse_transform = osr.CoordinateTransformation(new_cs,old_cs)

#get the point to transform, pixel (0,0) in this case
width = ds.RasterXSize
height = ds.RasterYSize
gt = ds.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width*gt[4] + height*gt[5]
maxx = gt[0] + width*gt[1] + height*gt[2]
maxy = gt[3]

#get the coordinates in lat long
latlong_min = transform.TransformPoint(minx,miny)
latlong_max = transform.TransformPoint(maxx,maxy)

# Manual input of OSM coord to check
osm_lat1 = 151.2061
osm_long1 = -33.8854
osm_lat2 = 151.2185
osm_long2 = -33.8956
back = inverse_transform.TransformPoint(latlong_min[0], latlong_min[1])
back1 = inverse_transform.TransformPoint(osm_lat1,osm_long1)  #these seem okay
back2 = inverse_transform.TransformPoint(osm_lat1,osm_long2) #these seem okay
back3 = inverse_transform.TransformPoint(osm_lat2,osm_long2) #these seem okay
back4 = inverse_transform.TransformPoint(osm_lat2,osm_long1) #these seem okay

# Convert world-coord to pixel coordinates
latLonPairs = [[back1[0], back1[1]], [back2[0],back2[1]],[back3[0],back3[1]],[back4[0],back4[1]]]
pixelPairs = []
for point in latLonPairs:
    x = (point[0]-gt[0])/gt[1]
    y = (point[1]-gt[3])/gt[5]
    # Add the point to our return array
    pixelPairs.append([int(x),int(y)])

#draw in image
corner1 = pixelPairs[0]
corner2 = pixelPairs[1]
corner3 = pixelPairs[2]
corner4 = pixelPairs[3]
im = cv2.imread('0_0_0_tex.tif')
cv2.namedWindow('map',cv2.WINDOW_NORMAL)
cv2.circle(im,(corner1[0],corner1[1]),50,(0,0,255),-1)
cv2.circle(im,(corner2[0],corner2[1]),50,(0,0,255),-1)
cv2.circle(im,(corner3[0],corner3[1]),50,(0,0,255),-1)
cv2.circle(im,(corner4[0],corner4[1]),50,(0,0,255),-1)
cv2.imshow('map', im)
cv2.waitKey()
cv2.destroyAllWindows()

# Create cut-out mask
mask = np.zeros(im.shape, dtype=np.uint8)
roi_corners = np.array([[corner1, corner3, corner4],[corner1, corner2, corner3]], dtype=np.int32)
white = (255, 255, 255)
cv2.fillPoly(mask, roi_corners, white)
# apply the mask
masked_image = cv2.bitwise_and(im, mask)
# display your handywork
cv2.namedWindow('masked image',cv2.WINDOW_NORMAL)
cv2.imshow('masked image', masked_image)
cv2.waitKey()
cv2.destroyAllWindows()
