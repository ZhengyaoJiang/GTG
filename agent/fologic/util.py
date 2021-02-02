from agent.fologic.base import *
from typing import List, Union

def duplicate_template(clause:Clause, times:int)->List[Clause]:
    clauses = []
    if times == 1:
        return [clause]
    for i in range(times):
        new_head = duplicate_atom(clause.head, str(i))
        new_body = [duplicate_atom(atom, str(i)) for atom in clause.body]
        clauses.append(Clause(new_head, new_body))
    return clauses

def duplicate_atom(atom:Atom, prefix:str)->Atom:
    pred = atom.predicate
    new_pred = Predicate(prefix+pred.name, pred.arity)
    return Atom(new_pred, atom.terms)