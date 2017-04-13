import pycatchmod
import click

@click.group()
@click.option("--debug/--no-debug", default=False)
@click.pass_context
def cli(ctx, debug):
    ctx.obj['debug'] = debug

def main():
    cli(obj={})

@cli.command()
@click.option("--filename", type=click.Path(exists=True), required=True)
@click.pass_context
def dump(ctx, filename):
    from pycatchmod.io.excel import open_workbook, read_parameters
    from json import dumps
    wb = open_workbook(filename)
    name, parameters = read_parameters(wb)
    print(dumps({"name": name, "subcatchments": parameters, "legacy": True}, indent=4, sort_keys=True))

@cli.command()
@click.option("--filename", type=click.Path(exists=True), required=True)
@click.option("--plot", default=False, is_flag=True)
@click.pass_context
def compare(ctx, filename, plot):
    from pycatchmod.io.excel import compare
    compare(filename, plot)

def date_parser(x):
    import pandas
    # try creating a date range based on the first date and the length
    dates = pandas.period_range(x[0], periods=len(x), freq="D")
    if pandas.Period(x.values[-1]) != dates[-1]:
        # parse dates one by one (much slower)
        # note this will assume american-style dates when faced with ambiguity
        dates = pandas.PeriodIndex(x.values, freq="D")
    return dates

def pandas_read(filename, key=None):
    import pandas  # TODO: move this somewhere else?
    if filename.endswith((".h5", ".hdf", ".hdf5")):
        df = pandas.read_hdf(filename, key=key)
    elif filename.endswith((".csv")):
        df = pandas.read_csv(filename)
        # parse dates
        df[df.columns[0]] = date_parser(df.loc[:, df.columns[0]])
        df.set_index(df.columns[0], inplace=True)
    return df

@cli.command()
@click.option("--parameters", type=click.Path(exists=True), required=True, help="Path to parameters JSON file")
@click.option("--rainfall", type=click.Path(exists=True), required=True, help="Path to input rainfall data")
@click.option("--rainfall-key", type=str, required=False, default=None, help="Key for input rainfall data (h5 files only)")
@click.option("--pet", type=click.Path(exists=True), required=True, help="Path to input PET data")
@click.option("--pet-key", type=str, required=False, default=None, help="Key for input PET data (h5 files only)")
@click.option("--output", type=click.Path(exists=None), required=True, help="Path to output file (with .csv or .h5 extension)")
@click.option("--output-key", type=str, default="flows", required=False, help="Key for output data (h5 files only)")
@click.option("--output-mode", type=click.Choice(["w", "a", "r+"]), default="a", required=False, help="Mode for output file (h5 files only)")
@click.option("--complib", type=click.Choice(["zlib", "bzip2", "lzo", "blosc", "none"]), required=False, default="zlib", help="Compression library (h5 files only)")
@click.option("--complevel", type=click.IntRange(1, 9), required=False, default=9, help="Compression level (h5 files only)")
@click.pass_context
def run(ctx, parameters, rainfall, pet, output, output_key, rainfall_key, pet_key, complib, complevel, output_mode):
    import pandas
    import numpy as np
    from pycatchmod.io.json import catchment_from_json
    from pycatchmod.io.excel import excel_parameter_adjustment
    from pycatchmod import run_catchmod

    print("Loading input data...")

    # load rainfall and pet data
    rainfall = pandas_read(rainfall, rainfall_key)
    pet = pandas_read(pet, pet_key)
    num_scenarios = rainfall.shape[1]

    # load catchmod model
    catchment = catchment_from_json(parameters, n=num_scenarios)
    excel_parameter_adjustment(catchment)

    idx0 = rainfall.index[0]
    idxN = rainfall.index[-1]
    dates = pandas.period_range(idx0, idxN, freq="D")

    # run catchmod
    print("Running catchmod...")
    flows = run_catchmod(catchment, rainfall.values, pet.values, dates=dates)
    print("Shape:", flows.shape)

    df = pandas.DataFrame(flows, index=dates)
    df.index.name = "Date"

    # write output
    if output.endswith((".h5", ".hdf", ".hdf5")):
        df.to_hdf(output, key=output_key, mode=output_mode, complib=complib, complevel=complevel)
    elif output.endswith(".csv"):
        # workaround for bug in pandas
        # see https://github.com/pandas-dev/pandas/issues/15982
        df.reset_index().to_csv(output, index=False)
    else:
        raise ValueError("Unrecognised output format.")

    print("Results:", output)

@cli.command()
def version():
    print("pycatchmod {}".format(pycatchmod.__version__))

if __name__ == "__main__":
    main()
