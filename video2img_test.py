from __future__ import print_function # must be the first
import os
#import gi
import sys
#gi.require_version('Gst', '1.0')
import cv2	#3.4.2.17
import math
import time
import json
import shutil
#import threading
import numpy as np
import multiprocessing as mp
from PIL import Image
#from gi.repository import Gst
from argparse import ArgumentParser
from pytesseract import image_to_string		#0.2.5





def rd(cap, duration=0, q=None):
	t0 = time.time()
	while True:
		ret, frame = cap.read()	#catch the picture from video
		q.put(frame)	#save video information in q
		time.sleep(0.025)	#reading interval
		
		if ((time.time()-t0)>=duration) or (frame is None):	#reading time of arrival
			break





def psnr(target, frame, height, PSNR_type=1):
	#identify the type of PSNR is RGB or YUV
	if PSNR_type==1:
		target = np.array(target, dtype=np.int)
		frame = np.array(frame,dtype=np.int)
		diff = target - frame
		mse = math.sqrt(np.mean(diff ** 2))	#Calculate MSE
		if mse<=0.0255:
			return 100
		else:
			return 20 * math.log10(255 / mse)
	else:
		#change RGB to YUV
		target = cv2.cvtColor(target, cv2.COLOR_BGR2YUV_I420)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

		#Calculate MSE
		seg1 = int(2*height/3)
		seg2 = int(3*height/4)
		target1 = np.array(target[0:seg1,:], dtype=np.int)
		target2 = np.array(target[seg1:seg2,:], dtype=np.int)
		target3 = np.array(target[seg2:height,:], dtype=np.int)
		frame1 = np.array(frame[0:seg1,:],dtype=np.int)
		frame2 = np.array(frame[seg1:seg2,:],dtype=np.int)
		frame3 = np.array(frame[seg2:height,:],dtype=np.int)
		diff1 = target1 - frame1
		diff2 = target2 - frame2
		diff3 = target3 - frame3
		mse1 = math.sqrt(np.mean(diff1 ** 2))
		mse2 = math.sqrt(np.mean(diff2 ** 2))
		mse3 = math.sqrt(np.mean(diff3 ** 2))
		
		#test whether PSNR large than 100
		if min(mse1,mse2,mse3)!=0:
			psnr1 = 20 * math.log10(255 / mse1)
			psnr2 = 20 * math.log10(255 / mse2)
			psnr3 = 20 * math.log10(255 / mse3)
			psnr = np.nanmean([psnr1,psnr2,psnr3])
			if psnr>=100:
				return 100
			else:
				return psnr
		else:
			return 100





def identify(golden_file, dstfile, path_f, q=None, q_out=None, width=1920, height=1080, PSNR_type=1, PSNR_LB=30, max_id=1, v0=0, v1=0, nummp=1):

	try:
		basename, ext = golden_file.split(".")
	except:
		output = {}
		output.update({'error': 'golden_file_type'})
		output = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
		print(output)		###
		path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=resfile)
		f = open(path_, 'w')
		f.write('測試項目		測試結果\n')
		f.write('error			比對檔案參數/圖片檔名設定錯誤')
		f.close()
		exit()


	low_wid = 0
	high_wid = int(0.1*width)
	low_ht = int(24.0/27*height)
	high_ht = int(51.0/54*height)

	#low_ht = int(53/60*height)		#720*480
	#high_ht = int(451/480*height)		#720*480
	name_ = []
	id_nb = []
	score = []
	head = -1
	tail = -1
	PSNR_low = []
	output_frame = []
	golden_frame = []


	print('ID identificating ...')
	while True:
		frame = q.get()	#get frames from reading before
		if frame is None:	#get all frames
			break		
		#test weather read error size
		while frame.shape != (height,width,3):
			frame = q.get()	#get frames from reading before
			if frame is None:	#get all frames
				break
		if frame is None:	#get all frames
			break
		output_frame.append(frame)	#save frames for outputing into pictures
		
		#ID identification
		id_1 = frame[low_ht:high_ht,low_wid:int(high_wid/2),:]
		id_1 = cv2.cvtColor(id_1, cv2.COLOR_BGR2GRAY)	#Gray scale
		id_1[id_1<130] = 0	#binarization
		id_1[id_1>=130] = 255
		id_1 = Image.fromarray(id_1)	#change to PIL.Image.Image form
		text1 = image_to_string(id_1, config='digits')	#identification
		id_2 = frame[low_ht:high_ht,int(high_wid/2):high_wid,:]
		id_2 = cv2.cvtColor(id_2, cv2.COLOR_BGR2GRAY)	#Gray scale
		id_2[id_2<130] = 0	#binarization
		id_2[id_2>=130] = 255
		id_2 = Image.fromarray(id_2)	#change to PIL.Image.Image form
		text2 = image_to_string(id_2, config='digits')	#identification
		text = text1 + text2
		text = ''.join(text.split())
		rep={'U':'0', 'D':'0', 'G':'0', 'I':'1', 'L':'1', ']':'1', '}':'1', 'Z':'2', '$':'3', '?':'7', 'T':'7', 'S':'8', 'Q':'9', '-':'', u'\u2019':'', "'":''}	#adjust terms
		text = text.strip()
		text = text.upper()
		for r in rep:	#Adjust identification result
			text = text.replace(r,rep[r])

		name_.append(0)
		golden_frame.append(0)
		id_nb.append(0)
		try:
			id_int = int(text)	#save id in int type
			id_str = str('%04d' % id_int)	#save id in str('%04d') type for filename
			filename = "{basename}{seq}.{ext}"\
				.format(basename=basename, seq=id_str, ext=ext)
			try:
				target = cv2.imread(filename)	#read golden sample
			except Exception:
				target = None
				pass

			#test whether the length of identified number is four 
			if (len(text)!=4)|(target is None):
				name_[len(name_)-1] = 'id_error'
				golden_frame[len(golden_frame)-1] = None
				id_nb[len(id_nb)-1] = -max_id-100
			else:
				tail = id_int
				if head == -1:
					head = id_int
				name_[len(name_)-1] = '{id_str}_{nb}'.format(id_str = id_str, nb = v1.value)
				v1.value += 1	#serial number
				golden_frame[len(golden_frame)-1] = target
				id_nb[len(id_nb)-1] = id_int				

		except Exception:
			name_[len(name_)-1] = 'id_error'
			golden_frame[len(golden_frame)-1] = None
			id_nb[len(id_nb)-1] = -max_id-100
			pass


	#test whether the test video repeat play
	if tail<=head:
		for i in range(len(id_nb)):
			if id_nb[i]<head:
				id_nb[i] += (max_id+1)


	#test whether id identify to other wrong number	
	for i in range(2,len(id_nb)-2):
		if (((id_nb[i]>=id_nb[i-2])+(id_nb[i]>=id_nb[i-1])+(id_nb[i]<=id_nb[i+1])+(id_nb[i]<=id_nb[i+2]))<3):
			#bug fix
			name_[i] = 'id_error'
			golden_frame[i] = None
			id_nb[i] = -max_id-100		

		if ((id_nb[i]-id_nb[i-2]<20)+(id_nb[i]-id_nb[i-1]<20)+(id_nb[i+1]-id_nb[i]<20)+(id_nb[i+2]-id_nb[i]<20)<2):
			#bug fix
			name_[i] = 'id_error'
			golden_frame[i] = None
			id_nb[i] = -max_id-100


	for i in range(len(id_nb)):
		if id_nb[i]>0:
			id_nb[i] %= (max_id+1)

	#calculation PSNR
	print('Calculating PSNR scores ...')

	#calculation PSNR
	for i in range(len(output_frame)):
		if golden_frame[i] is None:
			score.append(None)
		else:
			p = psnr(output_frame[i], golden_frame[i], height, PSNR_type)
			score.append(p)
			if p<PSNR_LB:	#test whether the PSNR of the picture is too low
				PSNR_low.append(name_[i])


	#output frames from video capture
	print('Outputting the pictures ...')
	for i in range(len(name_)):
		if name_[i]=='id_error':
			filename = "{source}/{seq}{nb}.png"\
				.format(source=path_f, seq=name_[i], nb=str('%04d' % v0.value))
			v0.value += 1
		else:
			filename = "{source}/img{seq}.png"\
				.format(source=path_f, seq=name_[i])

		try:
			cv2.imwrite(filename,output_frame[i])
		except Exception:
			pass


	q_out.put((PSNR_low, id_nb, name_, score, head, tail))





def video2img(srcfile=0, golden_file="", cur_time=None, duration=0.1, nummp=1, width=1920, height=1080, max_id=9999, PSNR_type=2, IDLB=0.7, PSNR_LB = 30.0, con_fail=5):

	#define folders and files for outputs
	if cur_time == None:
		cur_time = time.strftime('%Y%m%d_%H%M%S')

	outfile = "output_" + cur_time
	dstfile = "frames_" + cur_time	
	parfile = "parameter_" + cur_time
	PSNRfile = "PSNR_" + cur_time
	resfile = "result_" + cur_time


	#test whether dstfile exist
	path_f = "{source}/{folder}/{name}".format(source=os.getcwd(), folder=outfile, name=dstfile)
	if not os.path.exists(path_f):
		os.makedirs(path_f)

	if PSNR_type == 2:
		PSNR_type_str = 'YUV'
	else:
		PSNR_type_str = 'RGB'

	#output parameters
	path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=parfile)
	f = open(path_, 'w')
	f.write('輸入項目(參數名稱)				輸入內容\n')
	f.write('影片名稱(srcfile)				%s\n' %(srcfile) )
	f.write('比對資料夾名稱/檔案名稱(golden_file)		%s\n' %(golden_file) )
	f.write('比對檔案最大編號(max_id)			%s\n' %(max_id) )
	f.write('執行結果檔案夾名稱(outfile)			%s\n' %(outfile) )
	f.write('擷取影像結果檔案夾名稱(dstfile)			%s\n' %(dstfile) )
	f.write('PSNR數值檔案名稱(PSNRfile)			%s\n' %(PSNRfile) )
	f.write('參數紀錄檔案名稱(parfile)			%s\n' %(parfile) )
	f.write('測試結果檔案名稱(resfile)			%s\n' %(resfile) )
	f.write('測試時間(duration)				%s\n' %(duration) )
	f.write('執行核心數量(nummp)				%s\n' %	(nummp) )
	f.write('影像解析度寬(width)				%s\n' %(width) )
	f.write('影像解析度長(height)				%s\n' %(height) )
	f.write('影像格式(PSNR_type)				%s\n' %(PSNR_type_str) )
	f.write('判斷標準--成功率臨界值(IDLB)			%s\n' %(IDLB) )
	f.write('PSNR數值最低標準(PSNR_LB)			%s\n' %(PSNR_LB) )
	f.write('連續判斷失敗擷取影像數量(con_fail)		%s' %(con_fail) )
	f.close()


	print('Initializing ...')
	#test whether the parameter duration is negative number
	if duration < 0:
		output = {}
		output.update({'error': 'dur_non_positive'})
		output = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
		print(output)		###
		path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=resfile)
		f = open(path_, 'w')
		f.write('測試項目		測試結果\n')
		f.write('error			測試時間須大於0')
		f.close()
		exit()


	print('Reading the video ...') 
	#Start opencv to reading the video
	cap = cv2.VideoCapture(srcfile)	#Set the capture
	q = mp.Queue()	#Set queue for save pictures

	rd(cap, duration, q)	#catch all pictures in video
	cap.release()	#release the videocapture

	read_size = q.qsize()


	#test whether read anything
	if read_size==0:
		output = {}
		output.update({'error': 'read_non'})
		output = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
		print(output)		###
		path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=resfile)
		f = open(path_, 'w')
		f.write('測試項目		測試結果\n')
		f.write('error			未讀取到圖片')
		f.close()
		exit()


	#compare with golden sample
	for i in range(nummp):	#To end multiprocessing
		q.put(None)

	q_out = mp.Queue()
	v0 = mp.Value('i', 0)
	v1 = mp.Value('i', 0)

	for i in range(nummp):
		proc = mp.Process(target=identify,args=(golden_file, dstfile, path_f, q, q_out, width, height, PSNR_type, PSNR_LB, max_id, v0, v1, nummp))
		proc.start()	#start multiprocessing
	
	for i in range(nummp):
		proc.join()	#join multiprocessing

	#get information from identify function
	(PSNR_low, id_nb, name_, score, head, tail) = q_out.get()	
	for i in range(nummp-1):
		(new_PSNR_low, new_id_nb, new_name_, new_score, new_head, new_tail) = q_out.get()
		PSNR_low += new_PSNR_low
		id_nb += new_id_nb
		name_ += new_name_
		score += new_score
		head = min(head,new_head)
		tail = max(tail,new_tail)


	#test whether all scores is None
	print('Outputting the results ...')
	if head == -1:
		output = {}
		output.update({'error': 'non_identify'})
		output = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
		print(output)		###
		path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=resfile)
		f = open(path_, 'w')
		f.write('測試項目		測試結果\n')
		f.write('error			所有圖片辨識失敗')
		f.close()
		exit()


	for i in range(score.count(None)): score.remove(None)
	
	frames_loss = []
	id_nb_sort = id_nb.copy()
	id_nb_sort = list(set(id_nb_sort))
	id_nb_sort.sort()
	for i in range(len([i for i in id_nb_sort if i<0])):
		id_nb_sort.remove(id_nb_sort[0])

	#test which frames are loss
	for i in range(len(id_nb_sort)-1):
		if id_nb_sort[i+1]-id_nb_sort[i]!=1:
			l = range(id_nb_sort[i]+1,id_nb_sort[i+1],1)
			l = list(l)
			#print(l)
			for j in range(len(l)):
				l[j] %= (max_id+1)
			l = list(l)
			frames_loss += l

	PSNR_low_sort = list(set(PSNR_low))
	PSNR_low_sort.sort()
	for i in range(len(PSNR_low_sort)):
		if int(PSNR_low_sort[i][0:4]) in id_nb_sort:
			id_nb_sort.remove(int(PSNR_low_sort[i][0:4]))
		if int(PSNR_low_sort[i][0:4]) == tail:
			id_nb_sort.append(tail)
	
	if tail<=head:
		for i in range(id_nb_sort.index(tail)+1):
			id_nb_sort[i] += (max_id+1)

	id_nb_sort.sort()	#for find which frames are fail

	#test which frames are fail
	conti_frame_bad = []
	for i in range(len(id_nb_sort)-1):
		if id_nb_sort[i+1]-id_nb_sort[i]!=1:
			l = range(id_nb_sort[i]+1,id_nb_sort[i+1],1)
			l = list(l)
			for j in range(len(l)):
				l[j] %= (max_id+1)
			l = list(l)
			if len(l)>=con_fail:
				conti_frame_bad.append((id_nb_sort[i]+1,id_nb_sort[i+1]-1))


	id_rate = 100*(read_size-name_.count('id_error'))/read_size
	id_rate = float('%.2f' % id_rate)
	IDLB = 100*IDLB
	conti_fail_range = []

	#test which range are continuous fails
	for i in range(len(conti_frame_bad)):
		conti_fail_range.append('{0} to {1}'.format(conti_frame_bad[i][0] % (max_id+1),conti_frame_bad[i][1] % (max_id+1)))


	#output
	output = {}
	output.update({'basic_info': {'PSNR_mean': np.nanmean(score)}})
	if frames_loss!=[]:
		output['basic_info'].update({'frames_loss': frames_loss})
	if PSNR_low!=[]:
		output['basic_info'].update({'PSNR_low': PSNR_low_sort})
	if (id_rate>=IDLB)&(conti_frame_bad==[])&(np.nanmean(score)>=PSNR_LB):
		output.update({'result': 'Compliance'})
	else:
		output.update({'result': 'Non-compliance'})
		output.update({'fail_result': {}})
		if id_rate<IDLB:
			output['fail_result'].update({'identify_low': id_rate})
		if conti_frame_bad!=[]:
			output['fail_result'].update({'conti_frames_fail': conti_fail_range})
		if np.nanmean(score)<PSNR_LB:
			output['fail_result'].update({'PSNR_mean_low': np.nanmean(score)})
	output_basic = output['basic_info']
	output.pop('basic_info')
	output.update({'basic_info': output_basic})


	#output the output as result.txt
	path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=resfile)
	f = open(path_, 'w')
	f.write('測試項目			測試結果\n')

	for k, v in output.items():
		if isinstance(v,str):
			f.write('\n{1:<{0}s}				{3:^{2}s}\n'.format(len(k),k, len(v),v))

		else:
			f.write('\n{1:<{0}s}\n'.format(len(k),k))
			for k1, v1 in v.items():
				f.write('				%s : %s\n' %(k1,str(v1).strip("[]").strip("''").replace("'","")))
	f.close()

	output = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
	print(output)		###


	#output PSNR result
	path_ = "{source}/{folder}/{name}.txt".format(source=os.getcwd(), folder=outfile, name=PSNRfile)
	f = open(path_, 'w')
	f.write('Name	PSNR\n')
	for i in range(name_.count('id_error')):
		name_.remove('id_error')
	name_int = []
	for i in range(len(name_)):
		name_int.append(int(name_[i][0:4]))
	name_int_sort = name_int.copy()
	name_int_sort.sort()

	index_ = []
	index_.append(name_int.index(name_int_sort[0]))
	for i in range(1,len(name_int)):
		if name_int_sort[i]!=name_int_sort[i-1]:
			index_.append(name_int.index(name_int_sort[i]))
		else:
			index_.append(name_int.index(name_int_sort[i], index_[i-1]+1))

	for i in index_:
		f.write('{0}  {1}\n'.format(name_[i],'%.2f' % score[i]))
	f.close()





"""
if __name__=='__main__':
	t = time.time()
	#main program
	try:
		video2img(srcfile='big_buck_bunny/break_test_1080p.mp4', golden_file='big_buck_bunny/gold_1080p/img.png', cur_time=None, duration=5, nummp=4,width=1920, height=1080, max_id=9998, PSNR_type=2, IDLB=0.8, PSNR_LB=30, con_fail=5)
	except Exception:
		output = {}
		output.update({'error': 'execution_fail'})
		output = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
		print(output)		###
		exit()
	print("time:",time.time()-t)

"""





parser = ArgumentParser(prog = "video2img", usage = "Follow the descriptions of parameters and key in the parameters",
					description = "It will capture image from your connecting device. Compare with golden sample and test there are similar or not",
					epilog = "If it can't run,check your parameters and try again,please.")
parser.add_argument('srcfile', help = 'Input the name of the source video file')
parser.add_argument('golden_file', help = "Input the name of golden samples without number \nE.g. If name=gold0001.png in 'golden_sample' file, then input: golden_sample\gold.png")
parser.add_argument('-cur_time','-time', help = 'Input current time as serial number(default:None)', type = str, dest = 'cur_time', default = None)
parser.add_argument('-dur','-duration', help = 'Input how long you want to test(default:5)', type = float, dest = 'duration', default = 5)
parser.add_argument('-np','-nummp', help = 'Input how many core in your computer(default:1)', type = int, dest = 'nummp', default = 1)
parser.add_argument('-w','-width', help = 'Input the width of the frames(default:1920)', type = int, dest = 'width', default = 1920)
parser.add_argument('-ht','-height', help = 'Input the height of the frames(default:1080)', type = int, dest = 'height', default = 1080)
parser.add_argument('max_id', help = 'Input the max number of identity number', type = int)
parser.add_argument('-P_t','-PSNR_type', help = 'Input the PSNR type:1.RGB 2.YUV (default:YUV type)', metavar = 'PSNR_type', dest = 'PSNR_type', type = int,
					choices = range(1,3), default = 2)
parser.add_argument('-IDLB', help = 'Input the lower bound of the percent of identification rate(default:0.8)', type = float, default = 0.8)
parser.add_argument('-P_L','-PSNR_LB', help = 'Input the lower bound of the acceptable average of PSNR(default:30)', dest = 'PSNR_LB', type = float, default = 30)
parser.add_argument('-c_f','-con_fail', help = 'Input the max number of continuous frames identified fail(default:5)', type = int, dest = 'con_fail', default = 5)
args = parser.parse_args()





if __name__=='__main__':
	try:
		video2img(srcfile=args.srcfile, golden_file=args.golden_file, cur_time=args.cur_time, duration=args.duration, nummp=args.nummp,width=args.width, height=args.height, max_id=args.max_id, PSNR_type=args.PSNR_type, IDLB=args.IDLB, PSNR_LB=args.PSNR_LB, con_fail=args.con_fail)
	except Exception:
		output = {}
		output.update({'error': 'execution_fail'})
		output1 = json.dumps(output, indent = 25, separators = ( ',' , ':' ))
		print(output)		###
		exit()


