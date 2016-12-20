import xlrd
import numpy as np
import warnings

def open_workbook(filename):
    wb = xlrd.open_workbook(filename, on_demand=True)
    return wb

def read_parameters(wb):
    """Read catchmod model parameters

    Parameters
    ----------
    wb : xlrd.book.Book

    Returns
    -------
    name : string
    subcatchment : list of dictionaries

    TODO
    """
    sh = wb.sheet_by_name("Input Parameters")
    # perform some checks
    assert(sh.cell(5, 1).value == "Area Type")
    assert(sh.cell(17, 1).value == "Q1")
    # get number of areas
    # the excel model uses this number, even if more areas have been entered
    # in the parameters table
    assert(sh.cell(32, 3).ctype == xlrd.XL_CELL_NUMBER)
    num_areas = int(sh.cell(32, 3).value)
    name = sh.cell(0, 10).value.strip()
    # read the parameters for each catchment
    subcatchments = []
    for area_id in range(0, num_areas):
        col_num = 8 + area_id
        area_name = sh.cell(5, col_num).value.strip()
        if not area_name:
            continue
        subcatchment = {}
        subcatchment["name"] = area_name
        subcatchment["gradient_drying_curve"] = float(sh.cell(6, col_num).value)
        subcatchment["potential_drying_constant"] = float(sh.cell(7, col_num).value)
        subcatchment["direct_percolation"] = float(sh.cell(8, col_num).value)
        subcatchment["initial_upper_deficit"] = float(sh.cell(9, col_num).value)
        subcatchment["initial_lower_deficit"] = float(sh.cell(10, col_num).value)
        subcatchment["area"] = float(sh.cell(13, col_num).value)
        subcatchment["linear_storage_constant"] = float(sh.cell(14, col_num).value)
        if subcatchment["linear_storage_constant"] == 0:
            subcatchment["linear_storage_constant"] = None
        subcatchment["nonlinear_storage_constant"] = float(sh.cell(15, col_num).value)
        subcatchment["initial_linear_outflow"] = float(sh.cell(16, col_num).value)
        subcatchment["initial_nonlinear_outflow"] = float(sh.cell(17, col_num).value)
        subcatchments.append(subcatchment)
    return name, subcatchments

def read_results(wb):
    """Read catchmod timeseries input/outputs

    Returns a dictionary of numpy arrays containing the input and output
    timeseries from the model. The dictionary includes:

        "dates" : the date of the timestep
        "rainfall" : input rainfall in millimetres
        "pet" : input PET in millimeters
        "recharge" : recharge in millimetres
        "smd_upper" : upper soil moisture deficit in millimetres
        "smd_lower" : lower soil moisture deficit in millimetres
        "flow_sim" : simulated flow in megalitres per day
        "flow_obs" : observed flow in megalitres per day

    If the Excel model returns the flows in cumecs they are automatically
    converted in megalitres per day.
    """
    # It's possible that an excel model can use different rainfall/PET
    # timeseries for different areas. In this case it's not possible to
    # extract the input data from the spreadsheet as only the first
    # rainfall/PET timeseries is recorder in the outputs.
    sh = wb.sheet_by_name("Input Parameters")
    assert(sh.cell(32, 3).ctype == xlrd.XL_CELL_NUMBER)
    num_areas = int(sh.cell(32, 3).value)
    for area_id in range(0, num_areas):
        rain_col = sh.cell(26, 8).value
        pet_col = sh.cell(27, 8).value
        if rain_col != 1:
            raise ValueError("Unable to read input rainfall as column is not 1 (area {})".format(area_id))
        if pet_col != 1:
            raise ValueError("Unable to read input PET as column is not 1 (area {})".format(area_id))

    sh = wb.sheet_by_name("Catchment Timeseries")
    # perform some checks
    assert(sh.cell(4, 1).value == "Date")
    assert(sh.cell(4, 10).value.strip() == "AE")

    # find the number of rows in the data
    row = 6
    datemode = wb.datemode
    while True:
        try:
            date_num = sh.cell(row, 1).value
        except IndexError:
            break
        if not date_num:
            break
        row += 1
    max_row = row
    length = max_row - 6

    # allocate memory
    dates = np.empty([length,], dtype="<M8[D]")
    recharge = np.empty([length,], dtype=np.float64)
    smd_upper = np.empty([length,], dtype=np.float64)
    smd_lower = np.empty([length,], dtype=np.float64)
    flow_sim = np.empty([length,], dtype=np.float64)
    rain = np.empty([length,], dtype=np.float64)
    pet = np.empty([length,], dtype=np.float64)
    flow_obs = np.empty([length,], dtype=np.float64)

    # read the results in, row by row
    for n in range(0, length):
        row = n + 6
        date_num = sh.cell(row, 1).value
        year, month, day, hour, minute, second = xlrd.xldate_as_tuple(date_num, datemode)
        date = np.datetime64("{:04d}-{:02d}-{:02d}".format(year, month, day))
        dates[n] = date
        recharge[n] = sh.cell(row, 2).value
        smd_upper[n] = sh.cell(row, 3).value
        smd_lower[n] = sh.cell(row, 4).value
        flow_sim[n] = sh.cell(row, 5).value
        rain[n] = sh.cell(row, 8).value
        if(sh.cell(row, 8).ctype != xlrd.XL_CELL_NUMBER):
            warnings.warn("Invalid rainfall data")
            rain[n] = 0
        pet[n] = sh.cell(row, 9).value
        if(sh.cell(row, 9).ctype != xlrd.XL_CELL_NUMBER):
            warnings.warn("Invalid PET data")
            pet[n] = 0
        if(sh.cell(row, 12).ctype != xlrd.XL_CELL_NUMBER):
            # missing data for observed flow (this is possible)
            flow_obs[n] = np.nan
        else:
            flow_obs[n] = sh.cell(row, 12).value
        row += 1

    # sometimes the results are given in cumecs instead of megalitres per day
    # here we convert, so we're always working in Ml/d
    units = sh.cell(5, 6).value.strip()
    if units == "Mld":
        pass # already in Mld
    elif units == "cumecs":
        print("Converting from cumecs to Ml/d")
        flow_sim *= 86.4
        flow_obs *= 86.4

    return {
        "dates": dates,
        "rainfall": rain,
        "pet": pet,
        "recharge": recharge,
        "smd_upper": smd_upper,
        "smd_lower": smd_lower,
        "flow_sim": flow_sim,
        "flow_obs": flow_obs
    }

def excel_parameter_adjustment(C):
    """Adjust the parameters from Excel for pycatchmod"""
    for subcatchment in C.subcatchments:
        # unit conversion for nonlinear store
        if subcatchment.nonlinear_store is not None:
            subcatchment.nonlinear_store.nonlinear_storage_constant *= 86.4 * subcatchment.area
            subcatchment.nonlinear_store.initial_outflow *= np.array([86.4])

        # copied from excel
        soil_store = subcatchment.soil_store
        if soil_store.gradient_drying_curve == 0 or soil_store.potential_drying_constant == 0:
            soil_store.gradient_drying_curve = 0.000001

def compare(filename, plot=True):
    from .json import catchment_from_json
    wb = open_workbook(filename)
    name, subcatchments = read_parameters(wb)
    results = read_results(wb)
    catchment = catchment_from_json({"name": name, "subcatchments": subcatchments, "legacy": True})

    print("Loaded {} subcatchments".format(len(subcatchments)))
    print("Loaded {} timesteps of results".format(len(results["rainfall"])))

    rainfall = results['rainfall']
    pet = results['pet']

    if len(rainfall.shape) == 1:
        rainfall = rainfall[:, np.newaxis]
        pet = pet[:, np.newaxis]

    N = 1
    excel_parameter_adjustment(catchment)

    from pycatchmod import run_catchmod
    flow = run_catchmod(catchment, rainfall, pet)
    flow = flow[:,0]

    difference = flow - results["flow_sim"]
    MRSE = np.mean(np.abs(difference))
    MAX = difference.max()
    print("Maximum error:", MAX)
    print("MRSE error:", MRSE)

    if plot:
        import pandas
        import matplotlib.pyplot as plt
        # build data frame from data
        df = pandas.DataFrame({"Excel": results["flow_sim"], "Python": flow, "Rainfall": results["rainfall"], "PET": results["pet"]}, index=results["dates"])
        df = df[["Rainfall", "PET", "Excel", "Python"]]  # reorder columns
        # create figure
        fig, axarr = plt.subplots(2, 1, sharex=True)
        df.loc[:, ("Rainfall", "PET")].plot(ax=axarr[0])
        df.loc[:, "Excel"].plot(style=".-b", ax=axarr[1])
        df.loc[:, "Python"].plot(style=".-g", ax=axarr[1])
        plt.legend(["Excel", "Python"])
        axarr[1].grid(True)

        plt.show()
