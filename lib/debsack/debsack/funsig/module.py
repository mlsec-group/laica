import logging

from debsack.core.modules import FunctionModule, extract_metadata
from dwarfutils.utils import type2str, iter_formal_parameters, iter_call_sites

logger = logging.getLogger(__name__)


def linkage_name(subprogram):
    full_metadata = extract_metadata(subprogram)
    data = {
        'name': full_metadata['name'],
        'offset': full_metadata['offset'],
        'cu_offset': full_metadata['cu_offset'],
        'id': full_metadata['id'],
    }
    return data


class FunctionSignatureModule(FunctionModule):

    def label(self, subprogram):

        prototype = {
            'name': subprogram.name,
            'return_type': type2str(subprogram.type) if subprogram.type else "void",
            'parameter_list': [
                {
                    'index': i,
                    'name': parameter.name,
                    'type': type2str(parameter.type)
                }
                for i, parameter in enumerate(iter_formal_parameters(subprogram), 1)
            ]
        }

        return prototype

    def extract_data(self, subprogram):

        def iter_calls():
            for call_site in iter_call_sites(subprogram):
                callee = call_site.get_callee()
                if callee is not None:
                    yield call_site, callee

        ops = self._ops(subprogram.low_pc, subprogram.high_pc)

        call_sites = [
            {
                'low_pc': call_site.low_pc,
                'callee': linkage_name(callee)
            }
            for call_site, callee in iter_calls()]
        return {'ops': ops.hex(), 'call_sites': call_sites}
