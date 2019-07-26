# -*- coding: UTF-8 -*-# -*- coding: UTF-8 -*-
import os
import random
import time
import zipfile
import subprocess
import json, csv, re
import numpy as np
import pandas as pd
from pandas import DataFrame
from os import listdir, walk
from os.path import isfile, isdir, join
from flask import Flask, request, session, redirect, url_for, render_template, flash, jsonify, send_from_directory, send_file
from flask_wtf import FlaskForm
from celery import Celery
from subprocess import check_output
from subprocess import Popen, PIPE, STDOUT
from wtforms import StringField, TextAreaField, DateField, PasswordField, BooleanField, SubmitField, RadioField, SelectField, SelectMultipleField, FloatField, IntegerField
from wtforms.validators import DataRequired, InputRequired, Optional, Length, NumberRange, Regexp


############################## configuration ##############################
app = Flask(__name__)
# modules involved in security need to be encrypted: session, flask_wtf, flask_image, cookies
app.config['SECRET_KEY'] = 'SECRET_KEY'

app.config['CELERY_BROKER_URL'] = 'amqp://guest:guest@localhost:5672//'
app.config['CELERY_RESULT_BACKEND'] = 'amqp://guest:guest@localhost:5672//'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


########################## Find the golden file ###########################
class Findfile():
    path = '/home/student'

    # Step 1: remark all possible gold folder 
    dirlist = []
    dirootlist = []
    File = []
    for root, dirs, files in walk(path):
        for dirname in dirs:
            if 'gold' in dirname:
                dirlist.append(dirname) # gold folder
                dirootlist.append(join(root,dirname)) # root for gold folder
                File.append(listdir(join(root,dirname))) # gold file

    # Step 2: check files in gold folder ---- confirm every file name identical
    goldfold = []
    goldroot = []
    goldfile = []
    goldmaxid = []
    for i in range(len(dirlist)): # ith gold folder
        for j in range(len(File[i])): # jth gold file in ith gold folder
            files = File[i][j]
            file_check = re.sub("\d+", "", files) # remove the id number of gold files
            if j >=2 :
                filelast = re.sub("\d+", "", File[i][j-1]) # img.png

                # 2-1 if there are wrong format, break the rule
                if file_check != filelast:
                    failgold = dirlist[i]
                    break

                # 2-1 if all correct, record the gold information
                else:
                    if j == max(range(len(File[i]))):
                        goldfold.append(dirlist[i]) # golden folder
                        goldroot.append(dirootlist[i]) # root for golden folder
                        goldfile.append(File[i]) # files full name in golden folder
                        goldfile2 = re.sub("\d+", "", File[i][j]) # files simple name in golden folder
                        goldmaxid.append(len(File[i])-1)


############################### Flask Form ################################
class Form(FlaskForm):
    gold = Findfile() # import gold folder

    name = StringField('*1. 影片名稱(Film Name):',
	validators=[DataRequired(message='*Film Name is required.'), 
        Length(2,64)],default='H264_1080p.mp4')

    dirnames = gold.goldfold # gold folders at a list
    dir_form = []
    for dirname in gold.goldfold:
        dir_form.append((dirname,dirname)) # set gold folder for SelectField form
    gd_folder = SelectField('*2. 比對資料夾名稱(Golden Folder Name):',
	validators=[DataRequired(message='*Golden Folder Name is required.')],
        choices = dir_form)

    test_time = FloatField('3. 測試時間(Test Time(second)):',
        validators=[
        NumberRange(0.1,1800,"The time should be between %(min)s and %(max)s")],
        default='5') 

    core_num = SelectField('4. 執行核心數量(Number of Execution Cores):',choices = [
	('1', '1'),('2', '2'),('3', '3'),('4', '4'),
	('5', '5'),('6', '6'),('7', '7'),('8', '8')], default='1') 

    res = SelectField('5. 影像解析度(Image Resolution):',
        choices = [
	('640x480','640x480'),('800X600','800X600'),('1024x768','1024x768'),
	('1024x800','1024x800'),('1280X720','1280X720'),('1280X800','1280X800'),
        ('1336x768','1336x768'),('1920X1080','1920X1080'),('1920X1440','1920X1440')], 
        default='1920X1080')

    PSNR_type = SelectField('6. 影像格式(Image Format):', 
        choices = [('RGB','RGB'),('YUV','YUV')],
	default='YUV')

    IDLB = FloatField('7. 判斷標準--成功率臨界值(Identification Rate--The Proportion of Image recognition success rate):',
        validators=[
        NumberRange(0,1,"The time should be between %(min)s and %(max)s")]
        , default='0.8')

    PSNR_LB = FloatField('8. PSNR數值最低標準(The Lower Bound of PSNR Value(dB)):',
        validators=[
        NumberRange(1,100,"The time should be between %(min)s and %(max)s")]
        , default='30')

    con_fail = IntegerField('9. 連續判斷失敗擷取影像數量(Number of Continuous Images Identified Fail):',
        validators=[
        NumberRange(1,25,"The time should be between %(min)s and %(max)s")]
        , default='5')

    submit = SubmitField('Submit')


############################# Background task #############################
@celery.task(bind=True)
def long_task(self, formData):

    # cmd(terminal) command
    cmd = "python3 video2img_test.py %s %s %s -cur_time %s -dur %s -np %s -w %s -ht %s -P_t %s -IDLB %s -P_L %s -c_f %s" % (formData['name'], formData['gd_folder_file'], str(formData['gd_maxid']), formData['cur_time'], str(formData['test_time']), str(formData['core_num']), str(formData['res_width']), str(formData['res_height']), str(formData['PSNR_type']), str(formData['IDLB']), str(formData['PSNR_LB']), str(formData['con_fail']))

    # using subprocess.Popen() to conduct extra file 
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    l = [] # a list to store output (avoid replicated output)
    out = () # a tuple to store result (avoid replace output)

    while True:
        # read by line (byte -> string)
        output = p.stdout.readline().strip().decode("utf-8")

        while True:
            # if output is replicate, read next line
            if output in l:
                output = p.stdout.readline().strip().decode("utf-8")
            else:
                break

        l.append(output) # store output

        # status and percentage
        if output == 'Initializing ...':
            j = 0

        elif output == 'Reading the video ...':
            j = 10

        elif output == 'ID identificating ...':
            j = 40

        elif output == 'Calculating PSNR scores ...':
            j = 60

        elif output == 'Outputting the pictures ...':
            j = 80

        elif output == 'Outputting the results ...':
            # read the remain lines, which is the result
            out = p.stdout.readlines()
            # bytes in list to string in list
            out = [x.decode('utf-8') for x in out]
            # list to string
            tmp_ot = ''
            for i in out:
                tmp_ot = tmp_ot + i
            out = tmp_ot
            break

        else: # get a error message(string)
            output = 'Outputting the results ...'
            out = p.stdout.readline().strip().decode("utf-8")
            #out = "{" + p.stdout.readline().strip().decode("utf-8") + "}" #*****
            break

        time.sleep(1)
        self.update_state(state='PROGRESS',
                          meta={'current': j, 'total': 100, 'status': output})

    return {'current': 100, 'total': 100, 'status': 'Task completed!',
            'result': output, 'out': out}


####################### Get the parameters and POST #######################
@app.route('/')
@app.route('/form', methods=['GET', 'POST']) 
def forms():

    # create forms
    form = Form()
    gold = Findfile()
    formData = {
        'name': None,
        'gd_folder_file': None,
        'gd_file': None,
        'gd_maxid': None,
        'cur_time': None, # +++++
        'outfile': None, 
        'dstfile': None, 
        'PSNR_file': None, 
        'parfile': None, # +++
        'resfile': None, # +++
        'test_time': None,
        'core_num': None,
        'res_width': None,
        'res_height': None,
        'PSNR_type': None,
        'IDLB': None,
        'PSNR_LB': None,
        'con_fail': None,
        'submit': None
    }

    # POST: store the formData and enter the calculating web 
    if request.method == 'POST' and form.validate_on_submit():

        # 01.transform 'PSNR_type': trun RGB, YUV into 1,2
        image_format = 2
        if form.PSNR_type.data == 'RGB':
            image_format = 1

        # 02.transform 'res_width':choose the former part of 'res'
        width = str.split(form.res.data,'X')[0]

        # 03.transform 'res_height':choose the later part of 'res'
        length = str.split(form.res.data,'X')[1]

        # 04.record final gold name and maxid from Findfile()
        for i in range(len(gold.goldfold)):
            if form.gd_folder.data == gold.goldfold[i]:
                maxid = gold.goldmaxid[i]
            goldfull = form.gd_folder.data + '/' + gold.goldfile2 # gold folder/img.png

        # 05.record the remaining data from 'class Form()'
        cur_time = time.strftime('%Y%m%d_%H%M%S') # +++
        # output folder and file are named by time (time of submit parameters)
        outfile = 'output_' + cur_time
        dstfile = 'frames_' + cur_time
        PSNR_file = 'PSNR_' + cur_time
        parfile = 'parameter_' + cur_time # +++
        resfile = 'result_' + cur_time # +++

        formData = {
        'name': form.name.data,
        'gd_folder_file': goldfull,
        'gd_maxid': maxid,
        'cur_time': cur_time, # current time # +++
        'outfile': outfile, # named by time
        'dstfile': dstfile, # named by time
        'PSNR_file': PSNR_file, # named by time
        'parfile': parfile, # named by time # +++
        'resfile': resfile, # named by time # +++
        'test_time': form.test_time.data,
        'core_num': form.core_num.data,
        'res_width': width,
        'res_height': length,
        'PSNR_type_str': form.PSNR_type.data, # +++
        'PSNR_type': image_format,
        'IDLB': form.IDLB.data,
        'PSNR_LB': form.PSNR_LB.data,
        'con_fail': form.con_fail.data,
        'submit': form.submit.data
        }

        session["formData"] = formData # store parameters
        return render_template('celerytask.html', formData = formData)

    # GET
    elif form.errors and 'formData' in session:
        del session['formData']

    return render_template('video_forms.html', form=form)


########################## Start background task ##########################
@app.route('/longtask', methods=['POST'])
def longtask():
    # get formData and conduct long task
    formData = session.get("formData")
    task = long_task.apply_async(args=[formData])

    # get global var: task, return to function result()
    global task_global
    task_global = task
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


########################## Accessing Task Status ##########################
@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = long_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


############################# Show the output #############################
def outname(keys): # output form's name
    out_name = []
    for key in keys:
        if "identify_low" in key:
            out_name.append("辨識率(低於臨界值)")
        elif "conti_frames_fail" in key:
            out_name.append("連續辨識失敗項目")
        elif "PSNR_mean" in key:
            out_name.append("PSNR平均值")
        elif "frames_loss" in key:
            out_name.append("辨識失敗項目")
        elif "PSNR_low" in key:
            out_name.append("低PSNR項目")
    return(out_name)


@app.route('/output')
def result():
    # get the output image in the image folder
    formData = session.get("formData")
    outfile = formData['outfile']
    dstfile = formData['dstfile']
    folder = os.path.join(outfile, dstfile)

    class Imgid():
        err_id = []
        err_path = []
        for root, dirs, files in walk(folder):
            for name in files:
                if 'err' in name:
                    err_id.append(name)
                    err_path.append(join(root, name))
        enum = int(len(err_id))

    out_str = task_global.get()['out'] # original output, string type
    print(out_str)
    out = eval(out_str) # turn string　type into dictionary type


    # type I output : error
    try:
        er_v = out["error"] # the fail reason
        if er_v == "read_non":
            er_v == "未讀取到圖片"
        elif er_v == "golden_file_type":
            er_v = "比對檔案參數/圖片檔名設定錯誤"
        elif er_v == "non_identify":
            er_v == "所有圖片辨識失敗"
        elif er_v == "execution_fail":
            er_v = "執行錯誤"
        er_n = "辨識失敗原因"
        n0 = 1

    except:
        er_v = 'NA'
        er_n = 'NA'
        n0 = 0

    # type II output : result
    try:
        res_v = out["result"] # Compliance/Non-compliance
        if res_v == "Compliance":
            res_v = "Compliance(合格)"
        elif res_v == "Non-compliance":
            res_v = "Non-compliance(未合格)"
        res_n = "整體結果"
        n0_2 = 1
    except:
        res_v = 'NA'
        res_n = 'NA'
        n0_2 = 0

    # type III output : fail_result
    try:
        out_k1 = list(out['fail_result'].keys()) # 'identify_low, conti_frames_fail, PSNR_mean'
        out_n1 = outname(out_k1)
        n1 = len(out_k1)
        out_ov = list(out['fail_result'].values())
        out_v1 = []
        for i in range(n1):
            out_v1.append(str(out_ov[i]).strip("[]").strip("'").replace("'", ""))

    except:
        out_v1 = 'NA'
        out_n1 = 'NA'
        n1 = 0

    # type IV output : basic_inof

    ## part I : 'PSNR_low'
    try:
        bas = out['basic_info']
        Plow = list(bas['PSNR_low']) # mark PSNR_low files

        Plow_v = [] # record 'PSNR_low'
        for i in range(len(Plow)): # turn PSNR_low files into full file name
            Plow_v.append('img'+str(Plow[i])+'.png')
        pnum=int(len(Plow))
        pn=1
        Plow_n = '低PSNR項目'


    except:
        Plow_v = 'NA'
        Plow_n = 'NA'
        pnum=0
        pn=0

    ## part II : 'PSNR_mean, frames_fail'
    try:
        bas = out['basic_info']
        while 'PSNR_low' in bas: # remove PSNR_low files from output IV
            bas.pop('PSNR_low')

        out_k2 = list(bas.keys()) # record  'PSNR_mean, frames_fail'
        out_n2 = outname(out_k2) # rename the output name
        n2 = len(out_k2)
        out_ov = list(bas.values())
        out_v2 = [] # remove other symbol from putput
        for i in range(n2):
            out_v2.append(str(out_ov[i]).strip("[]").strip("''"))

    except:
        out_v2 = 'NA'
        out_n2 = 'NA'
        n2 = 0

        
    # other output : get err img from output image folder
    img = Imgid()
    img.err_id.sort()
    if img.enum != 0:
        err_n = "辨識失敗項目"
        en=1
    else:
        err_n='NA'
        err_id='NA'
        enum=0
        en=0

    return(render_template('output.html', 
            out = out, er_n=er_n, er_v=er_v, n0=n0, 
            res_n=res_n, res_v=res_v, n0_2=n0_2,
            out_n2=out_n2, out_v2=out_v2, n2=n2, 
            out_n1=out_n1, out_v1=out_v1, n1=n1,
            Plow_n=Plow_n, Plow_v=Plow_v, pnum=pnum, pn=pn,
            err_n=err_n, err_id=img.err_id, enum=img.enum, en=en))

@app.route('/<filename>')
def imgshow(filename):
    formData = session.get("formData")
    outfile = formData['outfile']
    dstfile = formData['dstfile']
    folder = os.path.join(outfile, dstfile)
    return send_from_directory(folder, filename)


################### Zip the output and can be downloaded ##################
@app.route("/output-downloads/", methods=["GET"])
def get_download_zip():
    # Determine the folders to zip
    formData = session.get("formData")
    outfile = formData['outfile']
    path = os.path.join(os.getcwd(), outfile)
    print(path)

    # Determine the destination of zip file
    zf = zipfile.ZipFile('%s.zip' %(outfile), mode='w')

    # Reading and writing ZIP files
    def Achive_Folder_To_ZIP(sFilePath):
        original_path = os.getcwd() # store the original path
        os.chdir(sFilePath) # change the current path

        # zip files in target directory
        for root, folders, files in os.walk("./"):
            for sfile in files:
                aFile = os.path.join(root, sfile)
                print(aFile)
                zf.write(aFile) # write the files in the target
        zf.close() # zip ends
        os.chdir(original_path) # change to the original path

    try:
        Achive_Folder_To_ZIP(path)
    except:
        print("Error: no such file or directory '%s'" %(path))

    # page for download .zip file
    return send_file('%s.zip' %(outfile), as_attachment=True, attachment_filename='%s.zip' %(outfile))


if __name__ == '__main__':
    app.run(debug=True)

