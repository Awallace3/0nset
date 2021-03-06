# Python 3.8.2

from flask import Flask, redirect, url_for, render_template, request, session, flash, abort
# Flask 1.1.2
from datetime import timedelta
from werkzeug.utils import secure_filename # Werkzeug 1.0.1
import os
from flask import send_from_directory
import numpy as np # numpy 1.19.0
import operator
import functools
import matplotlib.pyplot as plt # matplotlib: 3.1.2 
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure 
import base64

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'csv', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "hehehe"
app.permanent_session_lifetime = timedelta(minutes=5)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        text1 = request.form.get('text1')
        text2 = request.form.get('text2')
        x_unit = request.form.get('x_unit')
        y_unit = request.form.get('y_unit')
        x_direction = request.form.get('x_direction')
        y_norm_not = request.form.get('y_norm_not')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)


        if file and allowed_file(file.filename) and text1:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename,
                                    text1=text1,
                                    text2=text2,
                                    x_unit=x_unit,
                                    y_unit=y_unit,
                                    x_direction=x_direction,
                                    y_norm_not=y_norm_not))
    return render_template("first.html")

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def uploaded_file(filename):
    
    x_unit = request.args.get('x_unit', None)
    xu_abrev = x_unit
    if x_unit == "nm":
        x_unit = "Wavelength (nm)"
    elif x_unit == "cm-1":
        x_unit = "Wavenumbers (cm-1)"
    elif x_unit == "eV":
        x_unit = "Electron Volts (eV)"
    
    y_norm_not = request.args.get('y_norm_not', None)

    y_unit = request.args.get('y_unit', None)

    def Convert(string): 
        li = list(string.split(",")) 
        return li 

    def conv_num(string): 
        li = list(string.split(" ")) 
        return li 
    
    def rmse(polynomial, actual):
        return np.sqrt(((polynomial - actual) ** 2).mean())

    def matplotlib_make_plot(name, arr_x, arr_y, xs, ys, scaling, x_unit=x_unit, y_unit=y_unit):
        if x_unit == "Wavelength (nm)":
            x_ticks = []
            if arr_x[0] < arr_x[-1]:
                x_limits = [arr_x[0], arr_x[-1]]
            else:
                x_limits = [arr_x[-1], arr_x[0]]
            for i in arr_x:
                if i % 50 == 0:
                    x_ticks.append(i)
            if len(x_ticks) > 16:
                tmp_ls = []
                for i in x_ticks:
                    if i % 100 == 0:
                        tmp_ls.append(i)

                x_ticks = tmp_ls

        elif x_unit == "Wavenumbers (cm-1)":
            x_ticks = []
            if arr_x[0] > arr_x[-1]:
                x_limits = [arr_x[0], arr_x[-1]]
            else:
                x_limits = [arr_x[-1], arr_x[0]]

            for i in arr_x:
                if i % 200 == 0:
                    x_ticks.append(i)

        elif x_unit == "Electron Volts (eV)":
            x_ticks = []

            for i in arr_x:
                if i % 1 == 0:
                    x_ticks.append(i)

        if y_unit == "abs":
            y_unit = "Absorbance"

        if y_norm_not == 'normalize':
            y_limit = [0, 0.2]
            y_ticks = [0, 0.20, 0.40, 0.60, 0.80, 1.00, 1.2]
        else:
            y_limit = [0, scaling + scaling*0.05]

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(name)
        axis.set_xlabel(x_unit)
        axis.set_ylabel(y_unit)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limit)
        axis.set_xticks(x_ticks)
        if y_norm_not == 'normalize':
            axis.set_yticks(y_ticks)
        axis.plot(arr_x, arr_y, "ko-", linewidth = 1, markersize = 0)
        axis_0 = fig.add_subplot(1, 1, 1)
        axis_0.plot(xs, ys, "bo-", linewidth = 1, markersize = 0)
        axis.plot()
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        plt.clf()

        return pngImageB64String

    def matplotlib_polynomial(name, arr_x, arr_y, xs, ys, og_arr_x, scaling, x_unit=x_unit, y_unit=y_unit, y_norm_not=y_norm_not):        
        if x_unit == "Wavelength (nm)":
            x_ticks = []

            if arr_x[0] < arr_x[-1]:
                x_limits = [og_arr_x[0], og_arr_x[-1]]
            else:
                x_limits = [og_arr_x[-1], og_arr_x[0]]
            for i in og_arr_x:
                if i % 50 == 0:
                    x_ticks.append(i)
            if len(x_ticks) > 16:
                tmp_ls = []
                for i in x_ticks:
                    if i % 100 == 0:
                        tmp_ls.append(i)
                x_ticks = tmp_ls

        elif x_unit == "Wavenumbers (cm-1)":
            x_ticks = []
            if arr_x[0] > arr_x[-1]:
                x_limits = [og_arr_x[0], og_arr_x[-1]]
            else:
                x_limits = [og_arr_x[-1], og_arr_x[0]]
            for i in og_arr_x:
                if i % 200 == 0:
                    x_ticks.append(i)

        elif x_unit == "Electron Volts (eV)":
            x_ticks = []
            if arr_x[0] < arr_x[-1]:
                x_limits = [og_arr_x[0], og_arr_x[-1]]
            else:
                x_limits = [og_arr_x[-1], og_arr_x[0]]
            for i in og_arr_x:
                if i % 1 == 0:
                    x_ticks.append(i)

        if y_unit == "abs":
            y_unit = "Absorbance"
        
        if y_norm_not == 'normalize':
            y_limit = [0, 0.2]
            y_ticks = [0, 0.20, 0.40, 0.60, 0.80, 1.00, 1.2]
        else:
            y_limit = [0, scaling + scaling*0.05]

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title(name)
        axis.set_xlabel(x_unit)
        axis.set_ylabel(y_unit)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limit)
        axis.set_xticks(x_ticks)
        if y_norm_not == 'normalize':
            axis.set_yticks(y_ticks)
        axis.plot(arr_x, arr_y, "ko-", linewidth = 1, markersize = 0)
        axis_0 = fig.add_subplot(1, 1, 1)
        axis_0.plot(xs, ys, "bo--", linewidth = 1, markersize = 0)
        axis.plot()
        
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        plt.clf()
        
        return pngImageB64String

    def plt_save_edited_fig(arr_x_2, arr_y_2, xs_2, ys_2, x_range_start_1, x_range_end_1, title_1, file_1, radio1, x_axis_1, y_axis_1, y_range_start_1, y_range_end_1, x_unit=x_unit, y_unit=y_unit):
        
        if x_axis_1 == "Use Above":
            x_axis_1 = x_unit
        if y_axis_1 == "Use Above":
            y_axis_1 = y_unit

        x1_ticks = []
        x_range_start_1 = int(x_range_start_1)
        x_range_end_1 = int(x_range_end_1)

        for i in range(x_range_start_1, x_range_end_1+1, 1):
            x1_ticks.append(i)

        if x_unit == "Wavelength (nm)":
            x_ticks = []
            for i in x1_ticks:
                if i % 25 == 0:
                    x_ticks.append(i)
            
            if len(x_ticks) > 16:
                tmp_ls = []
                for i in x_ticks:
                    if i % 50 == 0:
                        tmp_ls.append(i)
                x_ticks = tmp_ls
                
                if len(x_ticks) > 16:
                    tmp_ls = []
                    for i in x_ticks:
                        if i % 100 == 0:
                            tmp_ls.append(i)
                    x_ticks = tmp_ls

        elif x_unit == "Wavenumbers (cm-1)":
            x_ticks = []
            for i in x1_ticks:
                if i % 200 == 0:
                    x_ticks.append(i)

        elif x_unit == "Electron Volts (eV)":
            x_ticks = []
            for i in x1_ticks:
                if i % 1 == 0:
                    x_ticks.append(i)

        y_range = [float(y_range_start_1), float(y_range_end_1)]

        if y_unit == "abs":
            y_ticks = []
            for i in np.arange(y_range[0], y_range[1], 0.2):
                y_ticks.append(i)

        plt.figure()
        
        plt.xlabel(x_axis_1)
        plt.ylabel("Absorbance")
        plt.xticks(x_ticks)
        plt.yticks(y_ticks) 
        plt.ylim(y_range)
        plt.xlim([x1_ticks[0], x1_ticks[-1]])
        plt.title(title_1)
        plt.plot(arr_x_2, arr_y_2, "ko-", linewidth = 1, markersize = 0)
        if radio1 == 'radio_y':
            plt.plot(xs_2, ys_2, "bo-", linewidth = 1, markersize = 0)
            plt.savefig("../" + file_1 + ".pdf")
            plt.clf()
        if radio1 == 'radio_n':
            plt.savefig("../" + file_1 + ".pdf")
            plt.clf()

        redirect(url_for('upload_file'))
        return
    

    def second_deriv_zero(sec_1):
        x_1 = np.array([sec_1[:,0]])
        y_1 = np.array([sec_1[:,1]])
        x_flat_1 = x_1.flatten()
        y_flat_1 = y_1.flatten()

        z_1 = np.polyfit(x_flat_1, y_flat_1, 4)
        p = np.poly1d(z_1)

        nest = []
        for i in x_flat_1[:]:
            nest.append(p(i))
        new_y = np.array(nest)
        new_y_flat = new_y.flatten()

        dy = np.diff(new_y_flat) / np.diff(x_flat_1)

        x_flat_dx = x_flat_1[:-1]

        ddy = np.diff(dy) / np.diff(x_flat_dx)

        x_flat_ddx = x_flat_dx[:-1]

        array_2nd = np.stack((x_flat_ddx, ddy), axis=1)

        highest = -10
        for i in array_2nd[:, 1]:
                    if i > highest:
                        highest = i

                        if i > 0:
                            break

        imp_val = np.where(array_2nd[:,1] == highest)

        int_imp_val = functools.reduce(lambda sub, ele: sub *10 + ele, imp_val[0])
        
        x_val_imp_val = array_2nd[int_imp_val, 0]

        return x_val_imp_val

    def steepest_line_input(ind, array_xy, array_x, array_y, neb, x_direction, y_norm_not=y_norm_not):
        """ This will ensure that the most steep line is plotted over the range """
        start_end = neb[1] - neb[0]
        teb = neb[:]

        if x_direction == "increasing":
            teb[0] = teb[0] + ind*4
            teb[1] = teb[1] + ind*4
            if teb[0] < array_x[0] or teb[0] > array_x[-1]:
                teb[0] = array_x[0]
            
            if teb[1] > array_x[-1] or teb[1] < array_x[0]:
                teb[1] = array_x[-1]
            
            test1 = np.array(teb)

            test = np.where(array_x[:] == test1[0])
            start = functools.reduce(lambda sub, ele: sub *10 + ele, test[0])

            test = np.where(array_x[:] == test1[1])
            end = functools.reduce(lambda sub, ele: sub *10 + ele, test[0])

            pre_sec = array_xy[start:end, 0:2]

            pre_max = np.max(pre_sec[:,1], axis=0) 

            row_max = np.where(array_xy[:,1] == pre_max)
            
            if len(row_max[0]) > 1:
                row_max = row_max[0]
                int_row_max = row_max[0]
                
            else: 
                
                int_row_max = functools.reduce(lambda sub, ele: sub *10 + ele, row_max[0])

            sec_1 = array_xy[int_row_max  : , 0:2] 

            if len(sec_1[:,0]) < 20:
                return None, None, None, None, None, None, None, None

            row_max_2 = np.where(array_xy[:,1] == pre_max)
            
            if len(row_max_2[0]) > 1:
                row_max_2 = row_max_2[0]
                int_row_max_2 = row_max_2[0]
            else:
                int_row_max_2 = functools.reduce(lambda sub, ele: sub *10 + ele, row_max_2[0])

            test2 = second_deriv_zero(sec_1)
            
            test_2 = np.where(array_x[:] == test2)
            if len(test_2[0]) > 1:
                test_2 = test_2[0]
                end_2 = test_2[0]
            else:
                end_2 = functools.reduce(lambda sub, ele: sub *10 + ele, test_2[0])

            pre_sec_2 = array_xy[int_row_max_2:end_2 + 20, 0:2]

            test1 = [array_xy[int_row_max_2,0], array_xy[end_2,0] + 20]

            sec_1 = pre_sec_2[:, 0:2] 

            x_1 = np.array([sec_1[:,0]])
            y_1 = np.array([sec_1[:,1]])
            x_flat_1 = x_1.flatten()
            y_flat_1 = y_1.flatten()

            z_1 = np.polyfit(x_flat_1, y_flat_1, 4)
            p = np.poly1d(z_1)

            nest = []
            for i in x_flat_1[:]:
                nest.append(p(i))
            new_y = np.array(nest)
            new_y_flat = new_y.flatten()

            rms = rmse(new_y_flat, y_flat_1)

            dy = np.diff(new_y_flat) / np.diff(x_flat_1)

            x_flat_dx = x_flat_1[:-1]

            ddy = np.diff(dy) / np.diff(x_flat_dx)

            x_flat_ddx = x_flat_dx[:-1]

            array_2nd = np.stack((x_flat_ddx, ddy), axis=1)

            highest = -10
            for i in array_2nd[:, 1]:
                        if i > highest:
                            highest = i

                            if i > 0:
                                break

            imp_val = np.where(array_2nd[:,1] == highest)
            int_imp_val = functools.reduce(lambda sub, ele: sub *10 + ele, imp_val[0])

            array_1st = np.stack((x_flat_dx, dy), axis=1)
            x_val_imp_val = array_2nd[int_imp_val, 0]

            deriv_slope = np.where(array_1st[:,0] == x_val_imp_val)
            x_int_imp_val = functools.reduce(lambda sub, ele: sub *10 + ele, deriv_slope[0])


            slope = array_1st[x_int_imp_val,1]

            b_val = np.where(array_xy[:,0] == x_val_imp_val)
            int_b_val = functools.reduce(lambda sub, ele: sub *10 + ele, b_val[0])

            b_line = array_xy[int_b_val,1]

            slope = float(slope)
            b_line = float(b_line)

            y_1_intercept = (slope) * (-x_val_imp_val) + b_line

            return slope, y_1_intercept, int_row_max_2, end_2, rms, test1, int_b_val, start_end
        
        else:

            teb[0] = teb[0] - ind*4
            teb[1] = teb[1] - ind*4

            if teb[0] > array_x[0] or teb[0] < array_x[-1]:
                teb[0] = array_x[0]
            
            if teb[1] < array_x[-1] or teb[1] > array_x[0]:
                teb[1] = array_x[-1]

            test1 = np.array(teb)

            test = np.where(array_x[:] == test1[1])
            start = functools.reduce(lambda sub, ele: sub *10 + ele, test[0])

            test = np.where(array_x[:] == test1[0])
            end = functools.reduce(lambda sub, ele: sub *10 + ele, test[0])

            pre_sec = array_xy[start:end, 0:2]

            pre_max = np.max(pre_sec[:,1], axis=0) 

            row_max = np.where(array_xy[:,1] == pre_max)
            
            if len(row_max[0]) > 1:

                row_max = row_max[0]
                int_row_max = row_max[0]
                
            else: 
                
                int_row_max = functools.reduce(lambda sub, ele: sub *10 + ele, row_max[0])
            
            sec_1 = array_xy[int_row_max  : , 0:2] 

            if len(sec_1[:,0]) < 20:
                return None, None, None, None, None, None, None, None

            row_max_2 = np.where(array_xy[:,1] == pre_max)
            
            if len(row_max_2[0]) > 1:
                row_max_2 = row_max_2[0]
                int_row_max_2 = row_max_2[0]
            else:
                int_row_max_2 = functools.reduce(lambda sub, ele: sub *10 + ele, row_max_2[0])

            test2 = second_deriv_zero(sec_1)

            test_2 = np.where(array_x[:] == test2)
            if len(test_2[0]) > 1:
                test_2 = test_2[0]
                end_2 = test_2[0]
            else:
                end_2 = functools.reduce(lambda sub, ele: sub *10 + ele, test_2[0])

            pre_sec_2 = array_xy[int_row_max_2:end_2 + 20, 0:2]

            test1 = [array_xy[int_row_max_2,0], array_xy[end_2,0] + 20]

            sec_1 = pre_sec_2[:, 0:2] 

            x_1 = np.array([sec_1[:,0]])
            y_1 = np.array([sec_1[:,1]])
            x_flat_1 = x_1.flatten()
            y_flat_1 = y_1.flatten()

            z_1 = np.polyfit(x_flat_1, y_flat_1, 4)
            p = np.poly1d(z_1)

            nest = []
            for i in x_flat_1[:]:
                nest.append(p(i))
            new_y = np.array(nest)
            new_y_flat = new_y.flatten()

            rms = rmse(new_y_flat, y_flat_1)


            dy = np.diff(new_y_flat) / np.diff(x_flat_1)


            x_flat_dx = x_flat_1[:-1]

            ddy = np.diff(dy) / np.diff(x_flat_dx)


            x_flat_ddx = x_flat_dx[:-1]

            array_2nd = np.stack((x_flat_ddx, ddy), axis=1)

            highest = -10
            for i in array_2nd[:, 1]:
                        if i > highest:
                            highest = i

                            if i > 0:
                                break

            imp_val = np.where(array_2nd[:,1] == highest)
            int_imp_val = functools.reduce(lambda sub, ele: sub *10 + ele, imp_val[0])

            array_1st = np.stack((x_flat_dx, dy), axis=1)
            x_val_imp_val = array_2nd[int_imp_val, 0]

            deriv_slope = np.where(array_1st[:,0] == x_val_imp_val)
            x_int_imp_val = functools.reduce(lambda sub, ele: sub *10 + ele, deriv_slope[0])


            slope = array_1st[x_int_imp_val,1]

            b_val = np.where(array_xy[:,0] == x_val_imp_val)
            int_b_val = functools.reduce(lambda sub, ele: sub *10 + ele, b_val[0])


            b_line = array_xy[int_b_val,1]
            
            slope = float(slope)
            b_line = float(b_line)

            y_1_intercept = (slope) * (-x_val_imp_val) + b_line


            return slope, y_1_intercept, int_row_max_2, end_2, rms, test1, int_b_val, start_end

    def absorption_onset(input_array_x, input_array_y, text1, text2, x_direction, y_norm_not=y_norm_not):

        array = np.stack((input_array_x, input_array_y), axis=1)
        array = array.astype(float)

        max_y_4 = np.max(array[:,1], axis=0)

        scaling = max_y_4

        row_num_max_y = np.where(array[:,1] == max_y_4)
        if len(row_num_max_y[0]) > 1:
            row_num_max_y = row_num_max_y[0] 
            int_row_max = row_num_max_y[0]
        else:
            int_row_max = functools.reduce(lambda sub, ele: sub *10 + ele, row_num_max_y[0])

        input_array_y = input_array_y.astype(float)
        arr_y_pre_4 = input_array_y.tolist()

        max_y_4 = max_y_4.astype(float)
        arr_y_4 = []

        for item in arr_y_pre_4:
            k = item / max_y_4
            arr_y_4.append(float(k))

        array_x = array[:,0]
        array_y = np.array(arr_y_4)
        array_xy = np.stack((array_x, array_y), axis = 1)
        max_y_4 = np.max(array_xy[:,1], axis=0)


        if text1[0] == "0.0" or text2[0] == "0.0":
            try:
                neb = [float(array_xy[int_row_max, 0]), float(array_xy[int_row_max + 90,0])]
            except IndexError:
                neb = [float(array_xy[int_row_max, 0]), float(array_xy[int_row_max + 20,0])]

        else:
            
            neb = [float(text1[0]), float(text2[0])]
        
        ######################## Define for Varying Ranges ################

        neb_difference = neb[1] - neb[0]

        if neb_difference > 100:
            size = 10
        elif neb_difference > 50 and neb_difference < 101:
            size = 6
        elif neb_difference > 20 and neb_difference < 51:
            size = 4
        elif neb_difference > 10 and neb_difference < 21:
            size = 2
        else:
            size = 1

        slope_test = []
        y_intercepts = []
        int_row_max_ts = []
        end_ts = []
        rms_ts = []
        used_ranges = []
        int_b_val_ts = []
        second_deriv_not_found = False

        for i in range(size):
            slope_t, y_intercept_t, int_row_max_t, end_t, rms_t, test1, int_b_val_t, start_end = steepest_line_input(i, array_xy, array_x, array_y, neb, x_direction)
            
            if slope_t == None:
                second_deriv_not_found = True
                break
            

            slope_test.append(slope_t)
            y_intercepts.append(y_intercept_t)
            int_row_max_ts.append(int_row_max_t)
            end_ts.append(end_t)
            rms_ts.append(rms_t)
            used_ranges.append(test1)
            int_b_val_ts.append(int_b_val_t)

        if second_deriv_not_found == True:
            return None, None, None, None, None, None, None, None, None, None, None


        best_fitting_range = []

        ############## making more accurate based off how close the line matches region 
        ## look into finding the range from 1st derive equaling zero 
        #### would like more testing, but it seems to be a positive

        for i in range(len(slope_test)):
            inner = []
            near_inflec_x = array_x[int_b_val_ts[i] - 15 : int_b_val_ts[i] + 15]
            near_inflec_y = array_y[int_b_val_ts[i] - 15 : int_b_val_ts[i] + 15]
            for cnt, j in enumerate(near_inflec_x):
                inner.append(slope_test[i]*j + y_intercepts[i])
                
                if cnt == len(near_inflec_x) -1:
                    inner =  np.array(inner)
                    match = np.isclose(inner, near_inflec_y, rtol=1e-06, atol=1e-02) # absolute(a - b) <= (atol + rtol * absolute(b))
                    
                    best_fitting_range.append(sum(match) + -1.5 *1000*slope_test[i])

        fitting_max = np.argmax(best_fitting_range)

        slope = slope_test[fitting_max]
        y_1_intercept = y_intercepts[fitting_max]
        int_row_max_2 = int_row_max_ts[fitting_max]
        end_2 = end_ts[fitting_max]
        rms = rms_ts[fitting_max]
        test1 = used_ranges[fitting_max]
        
        if y_norm_not == 'given_y':
            slope = slope*scaling
            y_1_intercept = y_1_intercept*scaling

        line_formula = "y = " + str(slope) + "x + " + str(y_1_intercept)

        final_x_value = -y_1_intercept / slope
        final_x_value = round(final_x_value, 1)
        
        ys_4 = []
        xs_4 = []

        for i in array_x[:]:
            ys_4.append(slope*i + y_1_intercept)

        for i in array_x[:]:
            xs_4.append(i)

        # gets flat x and y arrays to plot polynomial
        sec_1 = array_xy[int_row_max_2:end_2 + 20, 0:2]

        x_1 = np.array([sec_1[:,0]])
        y_1 = np.array([sec_1[:,1]])
        x_flat_1 = x_1.flatten()
        y_flat_1 = y_1.flatten()

        z_1 = np.polyfit(x_flat_1, y_flat_1, 4)
        p = np.poly1d(z_1)

        nest = []
        for i in x_flat_1[:]:
            nest.append(p(i))
        new_y = np.array(nest)
        new_y_flat = new_y.flatten()

        if y_norm_not == 'given_y':
            for num, j in enumerate(new_y_flat):
                new_y_flat[num] = j * scaling
            
            for num, j in enumerate(array_y):
                array_y[num] = j * scaling

        return final_x_value, line_formula, x_flat_1, new_y_flat, xs_4, ys_4, array_x, array_y, test1, rms, max_y_4, scaling

    text1 = request.args.get('text1', None)
    text2 = request.args.get('text2', None)
    x_direction = request.args.get('x_direction', None)
    
    if len(text1) > 6 or text1.isnumeric() == False:
        text1 = "0.0"
    text1 = conv_num(text1)

    if len(text2) > 6 or text2.isnumeric() == False:
        text2 = "0.0"
    text2 = conv_num(text2)

    x_range_input = [text1[0], text2[0]]

    f=open(filename,'r')
    lines = f.readlines()
    f.close() 

    baseline_in = False

    baseline = "Baseline"
    rm_lines= []
    with open(filename) as search:
        for num, line in enumerate(search,1):
            if baseline in line:
                rm_lines.append(num)
                baseline_in = True


    if baseline_in == True:

        del lines[rm_lines[1]-2:]
        del lines[rm_lines[0]]
        del lines[rm_lines[0]-1]

    else:

        del lines[0:2]

    start_ls = []
    for i in lines:
        i = i.rstrip()
        k = Convert(i)
        start_ls.append(k)

    big_array = np.array(start_ls)

    if big_array[0,-1] == '':
        big_array = big_array[:,:-1]

    big_array = big_array.astype(float)

    plot_filename = str(filename)
    plot_filename = ''.join(plot_filename.split())
    plot_filename = plot_filename[:-4]

    t1_x = big_array[:,0]
    t1_y = big_array[:,1]

    if t1_x[0] > t1_x[2] and x_direction== 'increasing':
            t1_x_f = np.flip(t1_x.flatten())
            t1_y_f = np.flip(t1_y.flatten())
        
    elif t1_x[0] > t1_x[2] and x_direction == 'decreasing':
        t1_x_f = t1_x.flatten()
        t1_y_f = t1_y.flatten()
        
    elif t1_x[0] < t1_x[2] and x_direction == 'decreasing':
        t1_x_f = np.flip(t1_x.flatten())
        t1_y_f = np.flip(t1_y.flatten())

    else:
        t1_x_f = t1_x.flatten()
        t1_y_f = t1_y.flatten()

    if big_array.shape[1] >= 4:

        t2_x = big_array[:,2]
        t2_y = big_array[:,3]

        if t2_x[0] > t2_x[2] and x_direction== 'increasing':
                t2_x_f = np.flip(t2_x.flatten())
                t2_y_f = np.flip(t2_y.flatten())
            
        elif t2_x[0] > t2_x[2] and x_direction == 'decreasing':
            t2_x_f = t2_x.flatten()
            t2_y_f = t2_y.flatten()
            
        elif t2_x[0] < t2_x[2] and x_direction == 'decreasing':
            t2_x_f = np.flip(t2_x.flatten())
            t2_y_f = np.flip(t2_y.flatten())

        else:
            t2_x_f = t2_x.flatten()
            t2_y_f = t2_y.flatten()

    if big_array.shape[1] >= 6:
        t3_x = big_array[:,4]
        t3_y = big_array[:,5]

        if t3_x[0] > t3_x[2] and x_direction== 'increasing':
                t3_x_f = np.flip(t3_x.flatten())
                t3_y_f = np.flip(t3_y.flatten())
            
        elif t3_x[0] > t3_x[2] and x_direction == 'decreasing':
            t3_x_f = t3_x.flatten()
            t3_y_f = t3_y.flatten()
            
        elif t3_x[0] < t3_x[2] and x_direction == 'decreasing':
            t3_x_f = np.flip(t3_x.flatten())
            t3_y_f = np.flip(t3_y.flatten())
        else:
            t3_x_f = t3_x.flatten()
            t3_y_f = t3_y.flatten()

    if big_array.shape[1] >= 8:
        t4_x = big_array[:,6]
        t4_y = big_array[:,7]

        if t4_x[0] > t4_x[2] and x_direction== 'increasing':
            t4_x_f = np.flip(t4_x.flatten())
            t4_y_f = np.flip(t4_y.flatten())
        
        elif t4_x[0] > t4_x[2] and x_direction == 'decreasing':
            t4_x_f = t4_x.flatten()
            t4_y_f = t4_y.flatten()
            
        elif t4_x[0] < t4_x[2] and x_direction == 'decreasing':
            t4_x_f = np.flip(t4_x.flatten())
            t4_y_f = np.flip(t4_y.flatten())
        else:
            t4_x_f = t4_x.flatten()
            t4_y_f = t4_y.flatten()

    try:
        final_x_value_1, line_formula1, x_flat_1, new_y_flat_1, xs_1, ys_1, arr_x_1, arr_y_1, test_1, rms_1, max_y_1, scaling_1 = absorption_onset(t1_x_f, t1_y_f, text1, text2, x_direction)
        x_range_out_1 = [int(arr_x_1[0]), int(arr_x_1[-1])]
        image_1 = matplotlib_make_plot("Figure 1", arr_x_1, arr_y_1, xs_1, ys_1, scaling_1)
        image_2 = matplotlib_polynomial("Figure 1's Polynomial Fitting", x_flat_1, new_y_flat_1, xs_1, ys_1, arr_x_1, scaling_1)
        
    except (NameError, TypeError):
        final_x_value_2, line_formula2, x_flat_2, new_y_flat_2, xs_2, ys_2, arr_x_2, arr_y_2, test_2, rms_2, max_y_2 = None, None, None, None, None, None, None, None, None, None, None
        image_1 = None
        image_2 = None
        x_range_out_1 = [None, None]

    try:
        final_x_value_2, line_formula2, x_flat_2, new_y_flat_2, xs_2, ys_2, arr_x_2, arr_y_2, test_2, rms_2, max_y_2, scaling_2 = absorption_onset(t2_x_f, t2_y_f, text1, text2, x_direction)
        x_range_out_2 = [int(arr_x_2[0]), int(arr_x_2[-1])]
        image_3 = matplotlib_make_plot("Figure 2", arr_x_2, arr_y_2, xs_2, ys_2, scaling_2)
        image_4 = matplotlib_polynomial("Figure 2's Polynomial Fitting", x_flat_2, new_y_flat_2, xs_2, ys_2, arr_x_2, scaling_2)

    except NameError:
        final_x_value_2, line_formula2, x_flat_2, new_y_flat_2, xs_2, ys_2, arr_x_2, arr_y_2, test_2, rms_2, max_y_2 = None, None, None, None, None, None, None, None, None, None, None
        image_3 = None
        image_4 = None
        x_range_out_2 = [None, None]

    try:
        final_x_value_3, line_formula3, x_flat_3, new_y_flat_3, xs_3, ys_3, arr_x_3, arr_y_3, test_3, rms_3, max_y_3, scaling_3 = absorption_onset(t3_x_f, t3_y_f, text1, text2, x_direction)
        x_range_out_3 = [int(arr_x_3[0]), int(arr_x_3[-1])]
        image_5 = matplotlib_make_plot("Figure 3", arr_x_3, arr_y_3, xs_3, ys_3, scaling_3)
        image_6 = matplotlib_polynomial("Figure 3's Polynomial Fitting", x_flat_3, new_y_flat_3, xs_3, ys_3, arr_x_3, scaling_3)

    except NameError:
        final_x_value_3, line_formula3, x_flat_3, new_y_flat_3, xs_3, ys_3, arr_x_3, arr_y_3, test_3, rms_3, max_y_3 = None, None, None, None, None, None, None, None, None, None, None
        image_5 = None
        image_6 = None 
        x_range_out_3 = [None, None]

    try:

        final_x_value_4, line_formula4, x_flat_4, new_y_flat_4, xs_4, ys_4, arr_x_4, arr_y_4, test_4, rms_4, max_y_4, scaling_4 = absorption_onset(t4_x_f, t4_y_f, text1, text2, x_direction)
        x_range_out_4 = [int(arr_x_4[0]), int(arr_x_4[-1])]
        image_7 = matplotlib_make_plot("Figure 4", arr_x_4, arr_y_4, xs_4, ys_4, scaling_4)
        image_8 = matplotlib_polynomial("Figure 4's Polynomial Fitting", x_flat_4, new_y_flat_4, xs_4, ys_4, arr_x_4, scaling_4)

    except NameError:
        final_x_value_4, line_formula4, x_flat_4, new_y_flat_4, xs_4, ys_4, arr_x_4, arr_y_4, test_4, rms_4, max_y_4 = None, None, None, None, None, None, None, None, None, None, None
        image_7 = None
        image_8 = None
        x_range_out_4 = [None, None]
    
    if request.method == 'POST':
        
        x_range_start_1 = request.form.get('x_range_start_1')
        x_range_end_1 = request.form.get('x_range_end_1')
        title_1 = request.form.get('Title1')
        file_1 = request.form.get('file1')
        radio1 = request.form.get('radio1')
        x_axis_1 = request.form.get('x_axis_1')
        y_range_start_1 = request.form.get('y_range_start_1')
        y_range_end_1 = request.form.get('y_range_end_1')
        y_axis_1 = request.form.get('y_axis_1')

        if x_axis_1 == "Use Above":
            x_axis_1 = x_unit
        if y_axis_1 == "Use Above":
            y_axis_1 = x_unit

        if x_range_start_1 and x_range_end_1 and title_1 and file_1:

            plt_save_edited_fig(arr_x_1, arr_y_1, xs_1, ys_1, x_range_start_1, x_range_end_1, title_1, file_1, radio1, x_axis_1, y_axis_1, y_range_start_1, y_range_end_1, x_unit=x_unit)

        x_range_start_2 = request.form.get('x_range_start_2')
        x_range_end_2 = request.form.get('x_range_end_2')
        title_2 = request.form.get('Title2')
        file_2 = request.form.get('file2')
        radio2 = request.form.get('radio2')
        x_axis_2 = request.form.get('x_axis_2')
        y_range_start_2 = request.form.get('y_range_start_2')
        y_range_end_2 = request.form.get('y_range_end_2')
        y_axis_2 = request.form.get('y_axis_2')

        if x_axis_2 == "Use Above":
            x_axis_2 = x_unit

        if x_range_start_2 and x_range_end_2 and title_2 and file_2:

            plt_save_edited_fig(arr_x_2, arr_y_2, xs_2, ys_2, x_range_start_2, x_range_end_2, title_2, file_2, radio2, x_axis_2, y_axis_2, y_range_start_2, y_range_end_2, x_unit=x_unit)
        
        x_range_start_3 = request.form.get('x_range_start_3')
        x_range_end_3 = request.form.get('x_range_end_3')
        title_3 = request.form.get('Title3')
        file_3 = request.form.get('file3')
        radio3 = request.form.get('radio3')
        x_axis_3 = request.form.get('x_axis_3')
        y_range_start_3 = request.form.get('y_range_start_3')
        y_range_end_3 = request.form.get('y_range_end_3')
        y_axis_3 = request.form.get('y_axis_3')

        if x_axis_3 == "Use Above":
            x_axis_3 = x_unit
        if y_axis_3 == "Use Above":
            y_axis_3 = x_unit

        if x_range_start_3 and x_range_end_3 and title_3 and file_3:

            plt_save_edited_fig(arr_x_3, arr_y_3, xs_3, ys_3, x_range_start_3, x_range_end_3, title_3, file_3, radio3, x_axis_3, y_axis_3, y_range_start_3, y_range_end_3, x_unit=x_unit)

        x_range_start_4 = request.form.get('x_range_start_4')
        x_range_end_4 = request.form.get('x_range_end_4')
        title_4 = request.form.get('Title4')
        file_4 = request.form.get('file4')
        radio4 = request.form.get('radio4')
        x_axis_4 = request.form.get('x_axis_4')
        y_range_start_4 = request.form.get('y_range_start_4')
        y_range_end_4 = request.form.get('y_range_end_4')
        y_axis_4 = request.form.get('y_axis_4')

        if x_axis_4 == "Use Above":
            x_axis_4 = x_unit
        if y_axis_4 == "Use Above":
            y_axis_4 = x_unit

        if x_range_start_4 and x_range_end_4 and title_4 and file_4:

            plt_save_edited_fig(arr_x_4, arr_y_4, xs_4, ys_4, x_range_start_4, x_range_end_4, title_4, file_4, radio4, x_axis_4, y_axis_4, y_range_start_4, y_range_end_4, x_unit=x_unit)

    return render_template("final_x.html", x_direction=x_direction, 
                            plot_filename=plot_filename, x_range_out_1=x_range_out_1, x_range_out_2=x_range_out_2,
                             x_range_out_3=x_range_out_3, x_range_out_4=x_range_out_4, max_y_2=max_y_2, max_y_1=max_y_1,
                             max_y_3=max_y_3, max_y_4=max_y_4, rms_2=rms_2, rms_3=rms_3, rms_4=rms_4, 
                             text1=x_range_input, test_2=test_2, test_3=test_3, test_4=test_4, 
                             line_formula1=line_formula1, line_formula4=line_formula4, line_formula2=line_formula2, 
                             line_formula3=line_formula3, 
                             final_x_value_2 = final_x_value_2, 
                             final_x_value_3=final_x_value_3, final_x_value_4=final_x_value_4,
                             final_x_value_1=final_x_value_1, test_1=test_1, rms_1=rms_1,
                             image_1=image_1, image_2=image_2, image_3=image_3, image_4=image_4,
                             image_5=image_5, image_6=image_6, image_7=image_7, image_8=image_8,
                             xu_abrev=xu_abrev)

if __name__ == "__main__":
    app.secret_key = "hehehe"
    app.run(debug=True)