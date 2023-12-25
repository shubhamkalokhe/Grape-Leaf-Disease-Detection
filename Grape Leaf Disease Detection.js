var testClass = ee.FeatureCollection('users/kalokheshubham34/testing');
print(testClass);
var trainClass = ee.FeatureCollection('users/kalokheshubham34/training');
print(trainClass);

Map.addLayer(roi,{},'ROI');
Map.centerObject(roi, 7);
//Map.addLayer(sample,{},'sample points');

// Load Sentinel-1 C-band SAR Ground Range collection (log scale, VV, descending)
var collectionVV = ee.ImageCollection("COPERNICUS/S1_GRD")
.filter(ee.Filter.eq('instrumentMode', 'IW'))
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
.filterMetadata('resolution_meters', 'equals' , 10)
.filterBounds(roi)
.select('VV');
//print(collectionVV, 'Collection VV'); 

// Load Sentinel-1 C-band SAR Ground Range collection (log scale, VH, descending)
var collectionVH = ee.ImageCollection("COPERNICUS/S1_GRD")
.filter(ee.Filter.eq('instrumentMode', 'IW'))
.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
.filterMetadata('resolution_meters', 'equals' , 10)
.filterBounds(roi)
.select('VH');
//print(collectionVH, 'Collection VH');

//Filter by date
var SARVV = collectionVV.filterDate('2016-10-01', '2016-11-01').median();

var SARVH = collectionVH.filterDate('2016-10-01', '2016-11-01').median();


// Function to cloud mask from the pixel_qa band of Sentinel_2 data.
function masksenti(image) {
var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000).select("B[2,3,4,7,8]*")
  .copyProperties(image, ["system:time_start"]);

}

// Extract the images from the Sentinel_2 collection
var collection_senti = ee.ImageCollection("COPERNICUS/S2")
.filterDate('2016-01-01', '2016-02-01')
.filterBounds(roi)
.map(masksenti);
//print(collection_senti, 'Sentinel_2');


//Calculate NDVI and create an image that contains Sentinel bands and NDVI
var comp = collection_senti.mean();
var ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI');

//Calculate GNDVI and create an image
var gndvi=comp.normalizedDifference(['B8', 'B3']).rename('GNDVI')


//Calculate REHBI
var red=comp.select('B4').clip(roi);

var nir=comp.select('B8').clip(roi);

var rededge=comp.select('B7').clip(roi);

var rehbi=(( rededge.subtract(red)).multiply(8.85)).subtract(nir.subtract(red).multiply(5.9))
.rename("REHBI");

var composite = ee.Image.cat(comp,ndvi,gndvi,rehbi);


//Apply filter to reduce speckle
var SMOOTHING_RADIUS = 50;
var SARVV_filtered = SARVV.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters');
var SARVH_filtered = SARVH.focal_mean(SMOOTHING_RADIUS, 'circle', 'meters');

//Display the SAR filtered images
//Map.addLayer(SARVV_filtered, {min:-15,max:0}, 'SAR VV Filtered',0);
//Map.addLayer(SARVH_filtered, {min:-25,max:0}, 'SAR VH Filtered',0);

//Merge Feature Collections
var newfc = trainClass;

//Define the SAR bands to train your data
var final = ee.Image.cat(SARVV_filtered,SARVH_filtered);
var bands = ['VH','VV'];
var training = final.select(bands).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30 });
  
//Train the classifier
var classifier = ee.Classifier.smileRandomForest(10).train({
  features: training,
  classProperty: 'landcover',
  inputProperties: bands
});

//Run the Classification
var classified = final.select(bands).classify(classifier);

//Display the Classification
Map.addLayer(classified.clip(roi), 
{min: 0, max: 1, palette: ['green', 'yellow']},
'SAR Classification');

// Create a confusion matrix representing resubstitution accuracy.
print('RF- SAR error matrix: ', classifier.confusionMatrix());
print('RF- SAR accuracy: ', classifier.confusionMatrix().accuracy());

//Repeat for Sentinel 2
//Define the Sentinel 2 bands to train your data
var bands_senti = ['B2', 'B3', 'B4','B7', 'B8','NDVI','GNDVI' ];
var training_senti = composite.select(bands_senti).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30
});

//Train the classifier
var classifier_senti = ee.Classifier.smileRandomForest(10).train({
  features: training_senti,
  classProperty: 'landcover',
  inputProperties: bands_senti
});

//Run the Classification
var classified_senti = composite.select(bands_senti).classify(classifier_senti);

//Display the Classification
Map.addLayer(classified_senti.clip(roi), 
{min: 0, max: 1, palette: ['green', 'yellow']},
'Optical Classification');

// Create a confusion matrix representing resubstitution accuracy.
print('RF-Senti error matrix: ', classifier_senti.confusionMatrix());
print('RF-Senti accuracy: ', classifier_senti.confusionMatrix().accuracy());

//Define both optical and SAR to train your data
var opt_sar = ee.Image.cat(composite, SARVV_filtered,SARVH_filtered);
var bands_opt_sar = ['VH','VV','B2', 'B3', 'B4','B7', 'B8','NDVI','GNDVI'];
var training_opt_sar = opt_sar.select(bands_opt_sar).sampleRegions({
  collection: newfc,
  properties: ['landcover'],
  scale: 30 });

//Train the classifier
var classifier_opt_sar = ee.Classifier.smileRandomForest(10).train({
  features: training_opt_sar, 
  classProperty: 'landcover',
  inputProperties: bands_opt_sar 
});

//Run the Classification
var classifiedboth = opt_sar.select(bands_opt_sar).classify(classifier_opt_sar);

//Display the Classification
var mask_o = composite.select(0).neq(1000)
var mask_r = SARVV_filtered.neq(1000)
var mask = mask_r.updateMask(mask_o)
Map.addLayer(classifiedboth.updateMask(mask).clip(roi), 
{min: 0, max: 1, palette: ['green', 'yellow']},
'Optical/SAR Classification');

// Create a confusion matrix representing resubstitution accuracy.
print('RF-Opt/SAR error matrix: ', classifier_opt_sar.confusionMatrix());
print('RF-Opt/SAR accuracy: ', classifier_opt_sar.confusionMatrix().accuracy());

//Extract 
var extractSAR=ee.Image(1).mask(classified.eq(1));                  
var pixel_area = extractSAR.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), 
                 roi, 30, null, null, false, 1e25).get('constant');
pixel_area = ee.Number(pixel_area).divide(10000);
print(pixel_area, 'grape area HA-SAR')

var extractSenti=ee.Image(1).mask(classified_senti.eq(1)).clip(roi);                  
var pixel_area = extractSenti.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), 
                 roi, 30, null, null, false, 1e25).get('constant');
pixel_area = ee.Number(pixel_area).divide(10000);
print(pixel_area, 'grape area HA- Optical Sentinel')
Map.addLayer(extractSenti, {}, 'Optical_Grape');


var extractBoth=ee.Image(1).mask(classifiedboth.eq(1));                  
var pixel_area = extractBoth.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), 
                 roi, 30, null, null, false, 1e25).get('constant');
pixel_area = ee.Number(pixel_area).divide(10000);
print(pixel_area, 'grape area HA-Both')

// Export the image, specifying scale and region.
 Export.image.toDrive({
   image: classifiedboth,
   description: 'MSI_SAR',
   scale: 30,
   fileFormat: 'GeoTIFF',
   crs:'EPSG:4326',
   maxPixels:1e13
 });

var vector=extractBoth.clip(roi).reduceToVectors({
  geometry:roi,
  crs:'EPSG:4326',
  scale:30,
  geometryType:'polygon',
  labelProperty:'landcover',
  maxPixels:1e45
  
});


//Extract NDVI
var ndvi_extract=composite.select(['NDVI']).clip(vector)
Map.addLayer(vector,{color:'red'},'cliped')
Map.addLayer(ndvi_extract,{max:-1,min:1,palette: ['green', 'yellow']},'ndvi_extract')

//Training 
var unsup_n=ndvi_extract.sample({
  region:geometry,
  scale:30
});

//Cluster NDVI
var cluster_1=ee.Clusterer.wekaKMeans(3).train(unsup_n);

//Testing 
var unsupervised_1=ndvi_extract.cluster(cluster_1).clip(roi);

//Map.addLayer(ndvi_extract,{min:-1,max:1},'Ndvi');
Map.addLayer(unsupervised_1,{min:0,max:2,palette: ['green', 'yellow','blue']},'unsupervised_1')

//EXPORT
Export.image.toDrive({
   image: unsupervised_1,
   description: 'Unsupervised_1',
   scale: 30,
   fileFormat: 'GeoTIFF',
   crs:'EPSG:4326',
   maxPixels:1e13
 });

//Extract GNDVI
var gndvi_extract=composite.select(['GNDVI']).clip(vector)
Map.addLayer(gndvi_extract,{max:-1,min:1,palette: ['green', 'yellow']},'gndvi_extract')
//*****
var unsup_g=gndvi_extract.sample({
  region:geometry,
  scale:30
});


//Cluster GNDVI
var cluster_2=ee.Clusterer.wekaKMeans(3).train(unsup_g);
var unsupervised_2=gndvi_extract.cluster(cluster_2).clip(roi);

//Map.addLayer(gndvi_extract,{min:-1,max:1},'Gndvi');
Map.addLayer(unsupervised_2,{min:0,max:2,palette: ['green', 'yellow','blue']},'unsupervised_2')

//Export
Export.image.toDrive({
   image: unsupervised_2,
   description: 'Unsupervised_2',
   scale: 30,
   fileFormat: 'GeoTIFF',
   crs:'EPSG:4326',
   maxPixels:1e13
 });


//Extract REHBI
var rehbi_extract=composite.select(['REHBI']).clip(vector)
Map.addLayer(rehbi_extract,{max:-1,min:1,palette: ['green', 'yellow']},'rehbi_extract')
//*****
var unsup_r=rehbi_extract.sample({
  region:geometry,
  scale:30
});


//Cluster REHBI
var cluster_3=ee.Clusterer.wekaKMeans(3).train(unsup_r);
var unsupervised_3=rehbi_extract.cluster(cluster_3).clip(roi);

//Map.addLayer(rehbi_extract,{min:-1,max:1},'Rehbi');
Map.addLayer(unsupervised_3,{min:0,max:2,palette: ['green', 'yellow','blue']},'unsupervised_3')

//Export
Export.image.toDrive({
   image: unsupervised_3,
   description: 'Unsupervised_3',
   scale: 30,
   fileFormat: 'GeoTIFF',
   crs:'EPSG:4326',
   maxPixels:1e30
 });
