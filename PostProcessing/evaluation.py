import __init__
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from LesionFinding import LesionFinding
import sklearn.metrics as skl_metrics
import inspect
from copy import deepcopy
import math
from matplotlib.ticker import ScalarFormatter,LogFormatter,StrMethodFormatter,FixedFormatter
import visualize_image_mask as vim

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        # print(Pz)
        # print(len(vec))
        # print(math.floor(Pz*len(vec)))
        # print(vec[math.floor(Pz*len(vec))])
        sens_lb[i] = vec[int(math.floor(Pz*len(vec)))]
        sens_up[i] = vec[int(math.floor((1.0-Pz)*len(vec)))]

    return sens_mean,sens_lb,sens_up

def getlesion(annotation, header, state = ""):
	lesion = LesionFinding()
	lesion.x_min = annotation[header.index(xmin_label)]
	lesion.y_min = annotation[header.index(ymin_label)]
	lesion.x_max = annotation[header.index(xmax_label)]
	lesion.y_max = annotation[header.index(ymax_label)]
	
	if lesionType_label in header:
		lesion.lesionType = annotation[header.index(lesionType_label)]

	if CADProbability_label in header:
		lesion.CADprobability = annotation[header.index(CADProbability_label)]
	
	if lesion_label in header:
		lesion.id = annotation[header.index(lesion_label)]
	
	if not state == "":
		lesion.state = state

	return lesion


def collect(annotations_filename, partition=2):
	annotations = read_csv_gt(partition, annotations_filename) 


	# annotations = csvTools.readCSV(annotations_filename)
	seriesUIDs_csv = [x[0] for x in annotations]
	seriesUIDs = list(set(seriesUIDs_csv))

	allLesions = collectLesionAnnotations(annotations, seriesUIDs)
	
	return (allLesions, seriesUIDs)



def collectLesionAnnotations(annotations, seriesUIDs):
	annotations.insert(0,['uid', 'x_min', 'y_min', 'x_max', 'y_max', 'label', 'slice'])
	alllesions = {}
	lesionCount = 0
	lesionCountTotal = 0
	
	for seriesuid in seriesUIDs:
		print('adding lesion annotations: ' + seriesuid)
		
		lesions = []
		numberOfIncludedlesions = 0
		
		# add included findings
		header = annotations[0]
		for annotation in annotations[1:]:
			lesion_seriesuid = annotation[header.index(seriesuid_label)]
			
			if seriesuid == lesion_seriesuid:
				lesion = getlesion(annotation, header, state = "Included")
				lesions.append(lesion)
				numberOfIncludedlesions += 1

		alllesions[seriesuid] = lesions
		lesionCount += numberOfIncludedlesions
		lesionCountTotal += len(lesions)
	
	print('Total number of included lesion annotations: ' + str(lesionCount))
	print('Total number of lesion annotations: ' + str(lesionCountTotal))
	 
	return alllesions

def generateBootstrapSet(scanToCandidatesDict, FROCImList,i):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_index_im   = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

    return candidates, rand_index_im

def computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,FROCImList,excludeList,numberOfBootstrapSamples=1000, confidence = 0.95):

    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
    
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)
    
    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}

    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:,i:i+1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid],candidate),axis = 1)

    btpsamps = []
    for i in range(numberOfBootstrapSamples):
        # rand_index_im   = np.random.randint(imageLen, size=imageLen)
        print('computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples))
        # Generate a bootstrapped set
        btpsamp, randinteger = generateBootstrapSet(scanToCandidatesDict,FROCImList_np,i)
        btpsamps.append(btpsamp)

    for btpsamp in btpsamps:
        fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
    
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # compute mean and CI
    interp_sens2save = deepcopy(interp_sens)
    # interp_sens2save = interp_sens[:]

    sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)


    return all_fps, sens_mean, sens_lb, sens_up


def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates

    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local)
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local)
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): # Handle border case when there are no false positives and ROC analysis give nan values.
      print("WARNING, this system has no false positives..")
      fps = np.zeros(len(fpr))
    else:
      fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, alllesions, CADSystemName, maxNumberOfCADMarks=-1,
				performBootstrapping=False,numberOfBootstrapSamples=1000,confidence = 0.95):
	'''
	function to evaluate a CAD algorithm
	@param seriesUIDs: list of the seriesUIDs of the cases to be processed
	@param results_filename: file with results
	@param outputDir: output directory
	@param alllesions: dictionary with all lesion annotations of all cases, keys of the dictionary are the seriesuids
	@param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
	'''

	lesionOutputfile = open(os.path.join(outputDir,'CADAnalysis.txt'),'w')
	lesionOutputfile.write("\n")
	lesionOutputfile.write((60 * "*") + "\n")
	lesionOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
	lesionOutputfile.write((60 * "*") + "\n")
	lesionOutputfile.write("\n")

	# results = csvTools.readCSV(results_filename)
	results = read_csv_pred(results_filename)
	results.insert(0,['uid', 'x_min', 'y_min', 'x_max', 'y_max', 'scores'])


	allCandsCAD = {}
	
	for seriesuid in seriesUIDs:
		
		# collect candidates from result file
		lesions = {}
		header = results[0]
		# header = header[0].split('\t')  # add this line if separated by \t, remove it if columns can be seen
		# print(header)
		
		i = 0
		stupid_flag = False
		for result in results[1:]:
			# result = result[0].split('\t')  # add this line if separated by \t, remove it if columns can be seen
			# print(seriesuid_label)
			# print(result)
			# print(header.index(seriesuid_label))
			# input('...')
			lesion_seriesuid = result[header.index(seriesuid_label)]

			
			
			if seriesuid == lesion_seriesuid:
				# stupid_flag = True
				lesion = getlesion(result, header)
				# print(lesion_seriesuid)

				lesion.candidateID = i
				lesions[lesion.candidateID] = lesion
				i += 1
		# if(stupid_flag):
		#     print('number of cands for this scan: ')
		#     print(len(lesions.keys()))
		#     print(i)
		#     raw_input('check how many cands added for this patient')


		if (maxNumberOfCADMarks > 0):
			# number of CAD marks, only keep must suspicous marks

			if len(lesions.keys()) > maxNumberOfCADMarks:
				# make a list of all probabilities
				probs = []
				for keytemp, lesiontemp in lesions.items():
					probs.append(float(lesiontemp.CADprobability))
				# print(len(probs))
				# raw_input('probability length')
				probs.sort(reverse=True) # sort from large to small
				probThreshold = probs[maxNumberOfCADMarks]
				# print(probThreshold)
				# raw_input('this is the prob_threshold')
				lesions2 = {}
				nrlesions2 = 0
				for keytemp, lesiontemp in lesions.items():
					if nrlesions2 >= maxNumberOfCADMarks:
						break
					if float(lesiontemp.CADprobability) > probThreshold:
						lesions2[keytemp] = lesiontemp
						nrlesions2 += 1

				lesions = lesions2
		# if(stupid_flag):
		#     print('number of cands for this scan: ')
		#     print(len(lesions.keys()))
		# raw_input('now what?')

		
		print('adding candidates: ' + seriesuid)
		allCandsCAD[seriesuid] = lesions
	
	# open output files
	nodNoCandFile = open(os.path.join(outputDir, "lesionsWithoutCandidate_%s.txt" % CADSystemName), 'w')
	
	# --- iterate over all cases (seriesUIDs) and determine how
	# often a lesion annotation is not covered by a candidate

	# initialize some variables to be used in the loop
	candTPs = 0
	candFPs = 0
	candFNs = 0
	candTNs = 0
	totalNumberOfCands = 0
	totalNumberOflesions = 0
	doubleCandidatesIgnored = 0
	irrelevantCandidates = 0
	minProbValue = -1000000000.0 # minimum value of a float
	FROCGTList = []
	FROCProbList = []
	FPDivisorList = []
	excludeList = []
	FROCtolesionMap = []
	ignoredCADMarksList = []

	# -- loop over the cases
	for seriesuid in seriesUIDs:
		# get the candidates for this case
		try:
			candidates = allCandsCAD[seriesuid]
		except KeyError:
			candidates = {}

		# add to the total number of candidates
		totalNumberOfCands += len(candidates.keys())

		# make a copy in which items will be deleted
		candidates2 = candidates.copy()


		# get the lesion annotations on this case
		try:
			lesionAnnots = alllesions[seriesuid]
		except KeyError:
			lesionAnnots = []

		# - loop over the lesion annotations
		for lesionAnnot in lesionAnnots:
			# increment the number of lesions
			if lesionAnnot.state == "Included":
				totalNumberOflesions += 1

			# x = float(lesionAnnot.coordX)
			# y = float(lesionAnnot.coordY)
			# z = float(lesionAnnot.coordZ)

			x_min = float(lesionAnnot.x_min)
			y_min = float(lesionAnnot.y_min)
			x_max = float(lesionAnnot.x_max)
			y_max = float(lesionAnnot.y_max)
			truth_coord = [x_min, y_min, x_max, y_max]


			# 2. Check if the lesion annotation is covered by a candidate
			# A lesion is marked as detected when the center of mass of the candidate is within a distance R of
			# the center of the lesion. In order to ensure that the CAD mark is displayed within the lesion on the
			# CT scan, we set R to be the radius of the lesion size.
			# diameter = float(lesionAnnot.diameter_mm)
			# if diameter < 0.0:
			# 	diameter = 10.0
			# radiusSquared = pow((diameter / 2.0), 2.0)

			found = False
			lesionMatches = []
			for key, candidate in candidates.items():
				# x2 = float(candidate.coordX)
				# y2 = float(candidate.coordY)
				# z2 = float(candidate.coordZ)
				x2_min = float(candidate.x_min)
				y2_min = float(candidate.y_min)
				x2_max = float(candidate.x_max)
				y2_max = float(candidate.y_max)
				candidate_coordinate = [x2_min, y2_min, x2_max, y2_max]
				# dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
				# if dist < radiusSquared:
				if criteria("IOU", truth_coord=truth_coord, pred_coord=candidate_coordinate, success_threshold=0.5):
					if (lesionAnnot.state == "Included"):
						found = True
						lesionMatches.append(candidate)
						if key not in candidates2.keys():

							print("This is strange: CAD mark with x_min of %s detected two lesions! Check for overlapping lesion annotations, SeriesUID: %s, lesion Annot ID: %s" % (str(candidate.x_min), seriesuid, str(lesionAnnot.id)))
						else:

							del candidates2[key]

					# elif (lesionAnnot.state == "Excluded"): # an excluded lesion
					# 	if bOtherlesionsAsIrrelevant: #    delete marks on excluded lesions so they don't count as false positives
					# 		if key in candidates2.keys():
					# 		  # print(key)
					# 		  # print(candidates2.keys())
					# 			irrelevantCandidates += 1
					# 			ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
					# 			del candidates2[key]
			
			if len(lesionMatches) > 1: # double detection
				doubleCandidatesIgnored += (len(lesionMatches) - 1)
			if lesionAnnot.state == "Included":
				# only include it for FROC analysis if it is included
				# otherwise, the candidate will not be counted as FP, but ignored in the
				# analysis since it has been deleted from the lesions2 vector of candidates
				if found == True:
					# append the sample with the highest probability for the FROC analysis
					maxProb = None
					for idx in range(len(lesionMatches)):
						# print(inspect.getmembers(candidate))
						# print('.........................')
						candidate = lesionMatches[idx]
						# # print(candidate)
						# print(inspect.getmembers(candidate))
						# input('check the candidate....')
						if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
							maxProb = float(candidate.CADprobability)

					FROCGTList.append(1.0)
					FROCProbList.append(float(maxProb))
					FPDivisorList.append(seriesuid)
					excludeList.append(False)
					FROCtolesionMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (seriesuid, lesionAnnot.id, lesionAnnot.x_min, lesionAnnot.y_min, lesionAnnot.x_max, float(lesionAnnot.y_max), str(candidate.id), float(candidate.CADprobability)))
					candTPs += 1
				else:
					print(lesionAnnot.x_min, lesionAnnot.y_min, lesionAnnot.seriesuid)
					print('########################################')
					candFNs += 1
					# append a positive sample with the lowest probability, such that this is added in the FROC analysis
					FROCGTList.append(1.0)
					FROCProbList.append(minProbValue)
					FPDivisorList.append(seriesuid)
					excludeList.append(True)
					FROCtolesionMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (seriesuid, lesionAnnot.id, lesionAnnot.x_min, lesionAnnot.y_min, lesionAnnot.x_max, float(lesionAnnot.y_max), int(-1), "NA"))
					nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (seriesuid, lesionAnnot.id, lesionAnnot.x_min, lesionAnnot.y_min, lesionAnnot.x_max, float(lesionAnnot.y_max), str(-1)))

		# add all false positives to the vectors
		for key, candidate3 in candidates2.items():
			candFPs += 1
			# print(key)
			# print(candidate3.CADprobability)
			# print(candidate3.seriesuid)
			# print(candidate3.coordX, candidate3.coordY, candidate3.coordZ)
			# raw_input('wait here!')
			FROCGTList.append(0.0)
			# print(inspect.getmembers(candidate3))
			# input('check the candidate....')
			FROCProbList.append(float(candidate3.CADprobability))
			FPDivisorList.append(seriesuid)
			excludeList.append(False)
			FROCtolesionMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate3.x_min, candidate3.y_min, candidate3.x_max, str(candidate3.id), float(candidate3.CADprobability)))

	if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(FROCtolesionMap) and len(FROCGTList) == len(excludeList)):
		lesionOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

	lesionOutputfile.write("Candidate detection results:\n")
	lesionOutputfile.write("    True positives: %d\n" % candTPs)
	lesionOutputfile.write("    False positives: %d\n" % candFPs)
	lesionOutputfile.write("    False negatives: %d\n" % candFNs)
	lesionOutputfile.write("    True negatives: %d\n" % candTNs)
	lesionOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)
	lesionOutputfile.write("    Total number of lesions: %d\n" % totalNumberOflesions)

	lesionOutputfile.write("    Ignored candidates on excluded lesions: %d\n" % irrelevantCandidates)
	lesionOutputfile.write("    Ignored candidates which were double detections on a lesion: %d\n" % doubleCandidatesIgnored)
	if int(totalNumberOflesions) == 0:
		lesionOutputfile.write("    Sensitivity: 0.0\n")
	else:
		lesionOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOflesions)))
	lesionOutputfile.write("    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))


	# # save for later use of FROC
	# np.save(output_address + 'For_FROC/FROCGTList',FROCGTList)
	# np.save(output_address + 'For_FROC/FROCProbList',FROCProbList)
	# np.save(output_address + 'For_FROC/seriesUIDs',seriesUIDs)
	# np.save(output_address + 'For_FROC/excludeList',excludeList)
	# np.save(output_address + 'For_FROC/FPDivisorList',FPDivisorList)
	# np.save(output_address + 'For_FROC/numberOfBootstrapSamples',numberOfBootstrapSamples)
	# np.save(output_address + 'For_FROC/confidence',confidence)



	# compute FROC
	fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)
	
	if performBootstrapping:
		fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,seriesUIDs,excludeList,
																  numberOfBootstrapSamples=numberOfBootstrapSamples, confidence = confidence)
		
	# Write FROC curve
	with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
		for i in range(len(sens)):
			f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
	
	# Write FROC vectors to disk as well
	with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
		for i in range(len(FROCGTList)):
			f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

	fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)
	
	sens_itp = np.interp(fps_itp, fps, sens)
	
	if performBootstrapping:
		# Write mean, lower, and upper bound curves to disk
		with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
			f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
			for i in range(len(fps_bs_itp)):
				f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
	else:
		fps_bs_itp = None
		sens_bs_mean = None
		sens_bs_lb = None
		sens_bs_up = None

	# create FROC graphs
	if int(totalNumberOflesions) > 0:
		graphTitle = str("")
		fig1 = plt.figure()
		ax = plt.gca()
		clr = 'b'
		# np.save(output_address + 'For_FROC/fps_itp',fps_itp)
		# np.save(output_address + 'For_FROC/sens_itp',sens_itp)
		# np.save(output_address + 'For_FROC/fps_bs_itp',fps_bs_itp)
		# np.save(output_address + 'For_FROC/sens_bs_mean',sens_bs_mean)
		# np.save(output_address + 'For_FROC/sens_bs_lb',sens_bs_lb)
		# np.save(output_address + 'For_FROC/sens_bs_up',sens_bs_up)

		plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
		if performBootstrapping:
			plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
			plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
			plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
			ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
		xmin = FROC_minX
		xmax = FROC_maxX
		plt.xlim(xmin, xmax)
		plt.ylim(0, 1)
		plt.xlabel('Average number of false positives per scan')
		plt.ylabel('Sensitivity')
		plt.legend(loc='lower right')
		plt.title('FROC performance - %s' % (CADSystemName))
		
		if bLogPlot:
			plt.xscale('log', base=2)
			ax.xaxis.set_major_formatter(FixedFormatter([1,2,4,8,16,32,64]))
		
		# set your ticks manually
		ax.xaxis.set_ticks([1,2,4,8,16,32,64])
		ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
		# plt.grid(b=True, which='both')
		plt.grid()
		plt.tight_layout()

		plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

	return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)



def FROC_Evaluation(gt_csv_dir, pred_csv_dir, log_dir, partition = 2):
	bPerformBootstrapping = True
	bNumberOfBootstrapSamples = 1000
	bConfidence = 0.95
	output_fileName = "detection.csv"

	results_filename = os.path.join(log_dir, output_fileName)

	(alllesions, seriesUIDs) = collect(gt_csv_dir, partition=2)
	evaluateCAD(seriesUIDs, pred_csv_dir, log_dir, alllesions,
				os.path.splitext(os.path.basename(results_filename))[0],
				maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
				numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

	# gt_list = read_csv_gt(partition=partition, csv_dir = gt_csv_dir)
	# pred_list = pd.read_csv(pred_csv_dir).values.tolist()


def visualize_directory(gt_csv_dir, pred_csv_dir, img_dir, partition, score_threshold):
	gt_list = read_csv_gt(partition=partition, csv_dir = gt_csv_dir)
	pred_list = pd.read_csv(pred_csv_dir).values.tolist()

	files = os.listdir(img_dir)
	for i, gt in enumerate(gt_list):
		volume = gt[0]
		found_files = [item for item in files if item.startswith(volume)]
		if not found_files[0].startswith('003171_05_01'):
			continue
		print(found_files)
		gt_bbox = [0,0,0,0]
		gt_bbox[0] = int(round(float(gt[1])))
		gt_bbox[1] = int(round(float(gt[2])))
		gt_bbox[2] = int(round(float(gt[3])))
		gt_bbox[3] = int(round(float(gt[4])))
		found_detections = [item for item in pred_list if item[0] == volume]
		print(gt[0])
		print(found_detections)
		pred_boxes = []
		for detection in found_detections:
		# for detection in found_detections[:1]:
			pred_box = [0, 0, 0, 0, 0] #x_min, y_min, x_max, y_max, score
			pred_box[0] = int(round(float(detection[1])))
			pred_box[1] = int(round(float(detection[2])))
			pred_box[2] = int(round(float(detection[3])))
			pred_box[3] = int(round(float(detection[4])))
			pred_box[4] = float(detection[5])

			if pred_box[4] >= score_threshold:
				pred_boxes.append(pred_box)


		img = np.load(os.path.join(img_dir, found_files[0]))
		vim.visual_eval(img, gt_bbox, pred_boxes)





def criteria(hit_def, truth_coord, pred_coord, success_threshold):
	if hit_def == "centroid":
		pred_center = [(pred_coord[0]+pred_coord[2])/2, (pred_coord[1]+pred_coord[3])/2]
		hit_flag = (truth_coord[0] <= pred_center[0]) and \
					(truth_coord[0] >= pred_center[2]) and \
					(truth_coord[1] >= pred_center[1]) and \
					(truth_coord[1] <= pred_center[3])
	if hit_def == 'IOU':
		assert truth_coord[0] < truth_coord[2]
		assert truth_coord[1] < truth_coord[3]
		assert pred_coord[0] < pred_coord[2]
		assert pred_coord[1] < pred_coord[3]

		# determine the coordinates of the intersection rectangle
		x_left = max(truth_coord[0], pred_coord[0])
		y_top = max(truth_coord[1], pred_coord[1])
		x_right = min(truth_coord[2], pred_coord[2])
		y_bottom = min(truth_coord[3], pred_coord[3])

		if x_right < x_left or y_bottom < y_top:
			return False

		# The intersection of two axis-aligned bounding boxes is always an
		# axis-aligned bounding box
		intersection_area = (x_right - x_left) * (y_bottom - y_top)

		# compute the area of both AABBs
		truth_coord_area = (truth_coord[2] - truth_coord[0]) * (truth_coord[3] - truth_coord[1])
		pred_coord_area = (pred_coord[2] - pred_coord[0]) * (pred_coord[3] - pred_coord[1])

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = intersection_area / float(truth_coord_area + pred_coord_area - intersection_area)
		assert iou >= 0.0
		assert iou <= 1.0
		hit_flag = iou >= success_threshold


		# xA = max(truth_coord[0], pred_coord[0])
		# yA = max(truth_coord[1], pred_coord[1])
		# xB = min(truth_coord[2], pred_coord[2])
		# yB = min(truth_coord[3], pred_coord[3])
		# interArea = (xB - xA) * (yB - yA)
		# boxAArea = (truth_coord[2] - truth_coord[0]) * (truth_coord[3] - truth_coord[1])
		# boxBArea = (pred_coord[2] - pred_coord[0]) * (pred_coord[3] - pred_coord[1])
		# iou = interArea / float(boxAArea + boxBArea - interArea)
		# hit_flag = iou >= success_threshold
	return hit_flag


def read_csv_gt(partition = 2, csv_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Documents_Information/DL_info.csv'):
	df = pd.read_csv(csv_dir)
	gt_list = []
	for idx, row in df.iterrows():
		volume_name = "_".join(row['File_name'].split("_")[:-1])
		volume_name = volume_name + '_' + str(row['Key_slice_index']).zfill(4)
		# volume_name = row['File_name'][:-4]
		slice_index = row['Key_slice_index']
		bbox = row['Bounding_boxes']
		lesion_type = row['Coarse_lesion_type']
		Possibly_noisy = row['Possibly_noisy']
		
		Image_size = row['Image_size']
		Train_Val_Test = row['Train_Val_Test']
		if Possibly_noisy == 1: 
			continue
		if Image_size.split(',')[0] != '512':
			continue

		if partition != Train_Val_Test: # pass partition = 2 for validation, and 3 for the final test
			continue
		
		y1 = float(bbox.split(',')[0])
		y2 = float(bbox.split(',')[2])
		x1 = float(bbox.split(',')[1])
		x2 = float(bbox.split(',')[3])
		gt_list.append([volume_name, y1, x1, y2, x2, lesion_type, slice_index])
	return gt_list


def read_csv_pred(csv_dir):
	df = pd.read_csv(csv_dir)
	pred_list = []
	for idx, row in df.iterrows():
		volume_name = row['uid']
		x_min = row['x_min']
		x_max = row['x_max']
		y_min = row['y_min']
		y_max = row['y_max']
		score = row['scores']
		pred_list.append([volume_name, x_min, y_min, x_max, y_max, score])
	
	return pred_list


if __name__ == "__main__":
	seriesuid_label = 'uid'
	xmin_label = 'x_min'
	ymin_label = 'y_min'
	xmax_label = 'x_max'
	ymax_label = 'y_max'
	lesionType_label = 'label'
	CADProbability_label = 'scores'
	lesion_label = 'slice'
	FROC_minX = 1 # Mininum value of x-axis of FROC curve
	FROC_maxX = 64 # Maximum value of x-axis of FROC curve
	bLogPlot = True
	gt_csv_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Documents_Information/DL_info.csv'
	pred_csv_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Models/2023-10-11_22-10-18.496446/detection.csv'
	log_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Models/2023-10-11_22-10-18.496446'
	FROC_Evaluation(gt_csv_dir, pred_csv_dir = pred_csv_dir, log_dir = log_dir, partition = 2)


	# # ######### Next lines for visual Evaluation #################
	# # print('came here')
	# gt_csv_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/Documents_Information/DL_info.csv'
	# pred_csv_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/Models/2023-10-11_22-10-18.496446/detection.csv'
	# img_dir = '/gpfs_projects/mohammadmeh.farhangi/DiskStation/DeepLesion/NIH/numpy_dataset/3_consecutive_slice/Tune'
	# visualize_directory(gt_csv_dir=gt_csv_dir, pred_csv_dir=pred_csv_dir, img_dir = img_dir, partition=2, score_threshold=0.0)