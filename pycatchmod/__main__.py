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
@click.option("--filename", type=str, required=True)
@click.pass_context
def dump(ctx, filename):
    from pycatchmod.io.excel import open_workbook, read_parameters
    from json import dumps
    wb = open_workbook(filename)
    name, parameters = read_parameters(wb)
    print(dumps({"name": name, "subcatchments": parameters}, indent=4, sort_keys=True))

@cli.command()
@click.option("--filename", type=str, required=True)
@click.option("--plot", default=False, is_flag=True)
@click.pass_context
def compare(ctx, filename, plot):
    from pycatchmod.io.excel import compare
    compare(filename, plot)

def pandas_read(filename):
    import pandas  # TODO: move this somewhere else?
    if filename.endswith((".h5", ".hdf", ".hdf5")):
        df = pandas.read_hdf(filename)
    elif filename.endswith((".csv")):
        df = pandas.read_csv(filename, parse_dates=True, dayfirst=True)
        df.set_index(df.columns[0], inplace=True)
    return df

@cli.command()
@click.option("--parameters", type=str)
@click.option("--rainfall", type=str)
@click.option("--pet", type=str)
@click.option("--output", type=str, required=True)
@click.pass_context
def run(ctx, parameters, rainfall, pet, output):
    import pandas
    import numpy as np
    from pycatchmod.io.json import catchment_from_json
    from pycatchmod import run_catchmod

    # load rainfall and pet data
    rainfall = pandas_read(rainfall)
    pet = pandas_read(pet)
    num_scenarios = rainfall.shape[1]

    # load catchmod model
    catchment = catchment_from_json(parameters, n=num_scenarios)

    idx0 = rainfall.index[0]
    idxN = rainfall.index[-1]
    dates = pandas.date_range(idx0, idxN, freq="D")

    # run catchmod
    flows = run_catchmod(catchment, rainfall.values, pet.values, dates=dates)
    print("Shape", flows.shape)

    df = pandas.DataFrame(flows, index=rainfall.index)
    df.index.name = "Date"

    # write output
    if output.endswith((".h5", ".hdf", ".hdf5")):
        df.to_hdf(output, key="flows", compress="zlib")
    elif output.endswith(".csv"):
        df.to_csv(output)
    else:
        raise ValueError("Unrecognised output format.")

    print("Results:", output)

@cli.command()
def version():
    print("pycatchmod", pycatchmod.__version__)

if __name__ == "__main__":
    main()
