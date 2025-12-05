#!/usr/bin/env python2
"""
Usage:
    generate_io_samples.py [options] PROGRAM_TEXT

Example:
    $ ./generate_io_samples.py "a <- [int] | b <- int | c <- TAKE b a | d <- COUNT isEVEN c | e <- TAKE d a"

Options:
    -h --help             Show this screen.
    -N --number NUM       Number of I/O examples to generate. [default: 5]
    -L --length NUM       Length of generated lists. [default: 10]
    -V --value-range NUM  Range of values. [default: 512]
"""

import os
import sys
from pathlib import Path
from collections import namedtuple, defaultdict
from math import ceil, sqrt

import numpy as np
import argparse


Function = namedtuple('Function', ['src', 'sig', 'fun', 'bounds'])
Program = namedtuple('Program', ['src', 'ins', 'out', 'fun', 'bounds'])


# HELPER FUNCTIONS
def type_to_string(t):
    if t == int:
        return 'int'
    if t == [int]:
        return '[int]'
    if t == bool:
        return 'bool'
    if t == [bool]:
        return '[bool]'
    raise ValueError('Type %s cannot be converted to string.' % t)


def scanl1(f, xs):
    if len(xs) > 0:
        r = xs[0]
        for i in range(len(xs)):
            if i > 0:
                r = f.fun(r, xs[i])
            yield r


##### Bound helpers:
def SQR_bounds(A, B):
    l = max(0, A)   # inclusive lower bound
    u = B - 1       # inclusive upper bound
    if l > u:
        return [(0, 0)]
    # now 0 <= l <= u
    # ceil(sqrt(l))
    # Assume that if anything is valid then 0 is valid
    return [(-int(sqrt(u)), ceil(sqrt(u+1)))]


def MUL_bounds(A, B):
    return SQR_bounds(0, min(-(A+1), B))


def scanl1_bounds(l, A, B, L):
    if l.src == '+' or l.src == '-':
        return [(A/L+1, B/L)]
    elif l.src == '*':
        return [(int((max(0, A)+1) ** (1.0 / L)), int((max(0, B)) ** (1.0 / L)))]
    elif l.src == 'MIN' or l.src == 'MAX':
        return [(A, B)]
    else:
        raise Exception('Unsupported SCANL1 lambda, cannot compute valid input bounds.')


##### LINQ language:
def get_language(V):
    Null = V
    lambdas = [
        Function('IDT',     (int, int),          lambda i: i,                                         lambda AB: [(AB[0], AB[1])]),

        Function('INC',     (int, int),          lambda i: i+1,                                       lambda AB: [(AB[0], AB[1]-1)]),
        Function('DEC',     (int, int),          lambda i: i-1,                                       lambda AB: [(AB[0]+1, AB[1])]),
        Function('SHL',     (int, int),          lambda i: i*2,                                       lambda AB: [((AB[0]+1)/2, AB[1]/2)]),
        Function('SHR',     (int, int),          lambda i: int(float(i)/2),                           lambda AB: [(2*AB[0], 2*AB[1])]),
        Function('doNEG',   (int, int),          lambda i: -i,                                        lambda AB: [(-AB[1]+1, -AB[0]+1)]),
        Function('MUL3',    (int, int),          lambda i: i*3,                                       lambda AB: [((AB[0]+2)/3, AB[1]/3)]),
        Function('DIV3',    (int, int),          lambda i: int(float(i)/3),                           lambda AB: [(AB[0], AB[1])]),

        Function('MUL4',    (int, int),          lambda i: i*4,                                       lambda AB: [((AB[0]+3)/4, AB[1]/4)]),
        Function('DIV4',    (int, int),          lambda i: int(float(i)/4),                           lambda AB: [(AB[0], AB[1])]),
        Function('SQR',     (int, int),          lambda i: i*i,                                       lambda AB: SQR_bounds(AB[0], AB[1])),
        #Function('SQRT',    (int, int),          lambda i: int(sqrt(i)),                              lambda (A, B): [(max(0, A*A), B*B)]),

        Function('isPOS',   (int, bool),         lambda i: i > 0,                                     lambda AB: [(AB[0], AB[1])]),
        Function('isNEG',   (int, bool),         lambda i: i < 0,                                     lambda AB: [(AB[0], AB[1])]),
        Function('isODD',   (int, bool),         lambda i: i % 2 == 1,                                lambda AB: [(AB[0], AB[1])]),
        Function('isEVEN',  (int, bool),         lambda i: i % 2 == 0,                                lambda AB: [(AB[0], AB[1])]),

        Function('+',       (int, int, int),     lambda i, j: i+j,                                    lambda AB: [(AB[0]/2+1, AB[1]/2)]),
        Function('-',       (int, int, int),     lambda i, j: i-j,                                    lambda AB: [(AB[0]/2+1, AB[1]/2)]),
        Function('*',       (int, int, int),     lambda i, j: i*j,                                    lambda AB: MUL_bounds(AB[0], AB[1])),
        Function('MIN',     (int, int, int),     lambda i, j: min(i, j),                              lambda AB: [(AB[0], AB[1])]),
        Function('MAX',     (int, int, int),     lambda i, j: max(i, j),                              lambda AB: [(AB[0], AB[1])]),
    ]

    LINQ = [
        Function('REVERSE', ([int], [int]),      lambda xs: list(reversed(xs)),                       lambda ABL: [(ABL[0], ABL[1])]),
        Function('SORT',    ([int], [int]),      lambda xs: sorted(xs),                               lambda ABL: [(ABL[0], ABL[1])]),
        Function('TAKE',    (int, [int], [int]), lambda n, xs: xs[:n],                                lambda ABL: [(0,ABL[2]), (ABL[0], ABL[1])]),
        Function('DROP',    (int, [int], [int]), lambda n, xs: xs[n:],                                lambda ABL: [(0,ABL[2]), (ABL[0], ABL[1])]),
        Function('ACCESS',  (int, [int], int),   lambda n, xs: xs[n] if n>=0 and len(xs)>n else Null, lambda ABL: [(0,ABL[2]), (ABL[0], ABL[1])]),
        Function('HEAD',    ([int], int),        lambda xs: xs[0] if len(xs)>0 else Null,             lambda ABL: [(ABL[0], ABL[1])]),
        Function('LAST',    ([int], int),        lambda xs: xs[-1] if len(xs)>0 else Null,            lambda ABL: [(ABL[0], ABL[1])]),
        Function('MINIMUM', ([int], int),        lambda xs: min(xs) if len(xs)>0 else Null,           lambda ABL: [(ABL[0], ABL[1])]),
        Function('MAXIMUM', ([int], int),        lambda xs: max(xs) if len(xs)>0 else Null,           lambda ABL: [(ABL[0], ABL[1])]),
        Function('SUM',     ([int], int),        lambda xs: sum(xs),                                  lambda ABL: [(ABL[0]/ABL[2]+1, ABL[1]/ABL[2])]),
    ] + \
    [Function(
            'MAP ' + l.src,
            ([int], [int]),
            lambda xs, l=l: list(map(l.fun, xs)),
            lambda AB, l=l: l.bounds((AB[0], AB[1]))
        ) for l in lambdas if l.sig==(int, int)] + \
    [Function(
            'FILTER ' + l.src,
            ([int], [int]),
            lambda xs, l=l: list(filter(l.fun, xs)),
            lambda AB, l=l: [(AB[0], AB[1])],
        ) for l in lambdas if l.sig==(int, bool)] + \
    [Function(
            'COUNT ' + l.src,
            ([int], int),
            lambda xs, l=l: len(list(filter(l.fun, xs))),
            lambda _, l=l: [(-V, V)],
        ) for l in lambdas if l.sig==(int, bool)] + \
    [Function(
            'ZIPWITH ' + l.src,
            ([int], [int], [int]),
            lambda xs, ys, l=l: [l.fun(x, y) for (x, y) in zip(xs, ys)],
            lambda AB, l=l: l.bounds((AB[0], AB[1])) + l.bounds((AB[0], AB[1])),
        ) for l in lambdas if l.sig==(int, int, int)] + \
    [Function(
            'SCANL1 ' + l.src,
            ([int], [int]),
            lambda xs, l=l: list(scanl1(l, xs)),
            lambda ABL, l=l: scanl1_bounds(l, ABL[0], ABL[1], ABL[2]),
        ) for l in lambdas if l.sig==(int, int, int)]

    return LINQ, lambdas

STRING_TO_TYPE = {
    'int': int,
    '[int]': [int]
}

def compile(source_code, LINQ, V, L, min_input_range_length=0):
    """ 
    Taken in a program source code, the integer range V and the tape lengths L,
    and produces a Program.
    If L is None then input constraints are not computed.
    """

    # used for lookup
    LINQ_names = [l.src for l in LINQ]

    # it is implied that for any k-th index for the lists below
    # always refer to the same line in the source code
    input_types = []
    types = []
    functions = []
    pointers = []
    
    for line in source_code.split('\n'):
        # get instruction part of statement
        instruction = line[5:]
        if instruction in STRING_TO_TYPE:
            # these are the inputs
            ins = STRING_TO_TYPE[instruction]
            input_types.append(ins)
            types.append(ins)
            functions.append(None)
            pointers.append(None)
        else:
            # all other non-input instructions processed here 

            split = instruction.split(' ')

            # the first word (space separated)
            command = split[0]

            # list of second word til last (space separated)
            args = split[1:]

            # restructure for higher order commands
            # NOTE this only works because of the limit to one alphabet vars
            if len(split[1]) > 1 or split[1] < 'a' or split[1] > 'z':
                command += ' ' + split[1]
                args = split[2:]

            # get function object based on name
            f = LINQ[LINQ_names.index(command)]
            assert len(f.sig) - 1 == len(args), "Wrong number of arguments for %s" % command

            # creates "pointers" which are unicode values of the variable letters
            ps = [ord(arg) - ord('a') for arg in args]

            # return type of this instruction
            # effectively type of variable on lhs
            types.append(f.sig[-1])

            # plumbing for executor
            # terrible
            functions.append(f)
            pointers.append(ps)

            # TODO might be expensive im removing because this shouldnt happen
            # assert [types[p] == t for p, t in zip(ps, f.sig)]

    # metadata
    # input_length: number of inputs
    # program_length: number of statements
    input_length = len(input_types)
    program_length = len(types)

    # Validate program by propagating input constraints and check all registers are useful
    limits = [(-V, V)]*program_length

    # TODO why on earth do we not do V bounds just because L is None?
    if L is not None:
        for t in range(program_length - 1, -1, -1):
            # not validating input setting lines
            if t >= input_length:
                lim_l, lim_u = limits[t]
                new_lims = functions[t].bounds((lim_l, lim_u, L))
                num_args = len(functions[t].sig) - 1
                for a in range(num_args):
                    p = pointers[t][a]
                    limits[pointers[t][a]] = (max(limits[p][0], new_lims[a][0]),
                                              min(limits[p][1], new_lims[a][1]))
                    #print('t=%d: New limit for %d is %s' % (t, p, limits[pointers[t][a]]))
            elif min_input_range_length >= limits[t][1] - limits[t][0]:
                # this check is done after the Vs are fully back propagated
                print('Program with no valid inputs: %s' % source_code)
                return None

    # for t in range(input_length, program_length):
    #     print('%s (%s)' % (functions[t].src, ' '.join([chr(ord('a') + p) for p in pointers[t]])))

    # Construct executor
    # NOTE these make shallow copies
    my_input_types = list(input_types)
    my_types = list(types)
    my_functions = list(functions)
    my_pointers = list(pointers)
    my_program_length = program_length

    def program_executor(args):
        # print '--->'
        # for t in range(input_length, my_program_length):
        #     print('%s <- %s (%s)' % (chr(ord('a') + t), my_functions[t].src, ' '.join([chr(ord('a') + p) for p in my_pointers[t]])))

        assert len(args) == len(my_input_types)
        registers = [None]*my_program_length

        # populates first n register for arguments
        for t in range(len(args)):
            registers[t] = args[t]

        # executes statements line by line
        for t in range(len(args), my_program_length):
            registers[t] = my_functions[t].fun(*[registers[p] for p in my_pointers[t]])

        # make executor return tuples
        output_value = registers[-1]
        try:
            output_value = tuple(output_value)
        except:
            # wasn't a list to begin with so we don't care
            ...

        # value found in last register is return value
        return output_value

    return Program(
        source_code,
        input_types,
        types[-1],
        program_executor,
        limits
    )

def generate_IO_examples(program, N, L, V):
    """ Given a programs, randomly generates N IO examples.
        using the specified length L for the input arrays. """
    input_types = program.ins
    input_nargs = len(input_types)

    # Generate N input-output pairs
    IO = []
    for _ in range(N):
        input_value = [None]*input_nargs
        for a in range(input_nargs):
            minv, maxv = program.bounds[a]
            if input_types[a] == int:
                input_value[a] = np.random.randint(minv, maxv)
            elif input_types[a] == [int]:
                input_value[a] = tuple(np.random.randint(minv, maxv, size=L))
            else:
                raise Exception("Unsupported input type " + input_types[a] + " for random input generation")
        
        output_value = program.fun(input_value)

        # change to tuple to use as key for dict
        input_value = tuple(input_value)

        IO.append((input_value, output_value))
        assert (program.out == int and output_value <= V) or (program.out == [int] and len(output_value) == 0) or (program.out == [int] and max(output_value) <= V)
    return IO


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(
        description = "Generates io samples for DeepCoder paper."
    )
    parser.add_argument("path", type=str, help="Name of file with program string.")
    parser.add_argument("-n", "--num", type=int, default=1)
    parser.add_argument("-l", "--length", type=int, default=10, help="Max length of a list.")
    parser.add_argument("-V", "--value_range", type=int, default=256, help="Value range of an integer such that -V <= i < V")
    parser.add_argument("--mode", choices=["EXAMPLES", "COMPILE"], default="EXAMPLES")

    args = parser.parse_args()

    prog_path = Path(args.path)

    N=args.num
    V=args.value_range
    L=args.length

    def print_int_or_list(var):
        if not isinstance(var, tuple):
            print(f"    {var}")
        else:
            print(f"    [", end="")
            print(f"{var[0]}", end="")
            if len(var) > 1:
                for p in var[1:]:
                    print(f", {p}",end="")
            print("]")

    match args.mode.lower():
        case "examples":
            try:
                with open(prog_path, "r") as f:
                    source = f.read()

                    print(f"generating {N} io samples with L={L}, V={V}")
                    print()
                    print(source)
                    print()

                    LINQ, _ = get_language(V)

                    source = source.replace(' | ', '\n')
                    program = compile(source, LINQ, V, L)

                    samples = generate_IO_examples(program, N, L, V)
                    for i, (inputs, output) in enumerate(samples):
                        print(f"SAMPLE {i}")
                        print(f"  INPUTS")
                        for input in inputs:
                            print_int_or_list(input)
                        
                        print(f"  OUTPUT")
                        print_int_or_list(output)
                    
                    print()
            except FileNotFoundError:
                print(f"Could not find file: {args.path}")
            except Exception as e:
                print(f"error! but i didn't expect this one!\n{e}")
        
        case "compile":
            try:
                with open(prog_path, "r") as f:
                    source = f.read()

                    print(f"compliling with L={L}, V={V}")
                    print()
                    print(source)
                    print()

                    LINQ, _ = get_language(V)

                    source = source.replace(' | ', '\n')
                    program = compile(source, LINQ, V, L)
                    
                    
                    inputs = [list(map(lambda x: int(x), list("100110101010001100111010")))]
                    output = program.fun(inputs)
                    print(f"  INPUTS")
                    for input in inputs:
                        print_int_or_list(input)
                    
                    print(f"  OUTPUT")
                    print_int_or_list(output)

            except FileNotFoundError:
                print(f"Could not find file: {args.path}")
            except Exception as e:
                print(f"error! but i didn't expect this one!\n{e}")

    
    


