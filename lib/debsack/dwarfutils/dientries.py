import os

from elftools.common.exceptions import DWARFError


def with_attr(attribute):
    def handler(attribute):

        def get_DIE(die):
            if attribute in die.attributes:
                return die
            if "DW_AT_abstract_origin" in die.attributes:
                return get_DIE(die.get_DIE_from_attribute('DW_AT_abstract_origin'))

        # Special handler for DW_AT_high_pc.
        # This attribute can be the high address or the size of the address range.
        if attribute == 'DW_AT_high_pc':
            def handler_(self):

                die = get_DIE(self._die)
                if die is None:
                    return None

                attr = die.attributes[attribute]
                if attr.form == 'DW_FORM_addr':
                    return attr.value
                else:
                    assert (attr.form == 'DW_FORM_data1'
                            or attr.form == 'DW_FORM_data2'
                            or attr.form == 'DW_FORM_data4'
                            or attr.form == 'DW_FORM_data8'
                            or attr.form == 'DW_FORM_sdata'
                            or attr.form == 'DW_FORM_udata'), f"Expected DW_FORM_data*, got {attr.form}"
                    return self.low_pc + attr.value
        else:
            def handler_(self):
                die = get_DIE(self._die)
                if die is None:
                    return None

                # If the target DIE is referencing another DIE, return this DIE.
                try:
                    return die.get_DIE_from_attribute(attribute)
                except NotImplementedError:
                    pass
                except DWARFError:
                    pass
                except TypeError:
                    pass

                # Return the (decoded) attribute value
                attr = die.attributes[attribute]
                try:
                    return attr.value.decode()
                except:
                    return attr.value

        return handler_

    class Mixin:
        pass

    setattr(Mixin, attribute[6:], property(handler(attribute)))

    return Mixin


class DIE:
    def __init__(self, die):
        assert type(self).tag == die.tag, f"expected {die.tag}"
        self._die = die

    def __getattr__(self, item):
        return getattr(self.__dict__['_die'], item)


class CompileUnitDIE(DIE,
                     with_attr("DW_AT_name"),
                     with_attr("DW_AT_language"),
                     with_attr("DW_AT_producer")):
    tag = 'DW_TAG_compile_unit'


class SubProgramDIE(DIE,
                    with_attr("DW_AT_name"),
                    with_attr("DW_AT_type"),
                    with_attr("DW_AT_abstract_origin"),
                    with_attr("DW_AT_external"),
                    with_attr("DW_AT_low_pc"),
                    with_attr("DW_AT_high_pc"),
                    with_attr("DW_AT_ranges"),
                    with_attr("DW_AT_decl_file"),
                    with_attr("DW_AT_decl_line"),
                    with_attr("DW_AT_decl_column"),
                    with_attr("DW_AT_external")):
    tag = 'DW_TAG_subprogram'

    @property
    def has_callsite(self):
        attr = self.attributes.get("DW_AT_GNU_all_call_sites", None)
        return attr.value if attr else False


class FormalParameterDIE(DIE,
                         with_attr("DW_AT_name"),
                         with_attr("DW_AT_location"),
                         with_attr("DW_AT_type")):
    tag = 'DW_TAG_formal_parameter'


class VariableDIE(DIE,
                  with_attr("DW_AT_name"),
                  with_attr("DW_AT_location"),
                  with_attr("DW_AT_type")):
    tag = 'DW_TAG_variable'


class GNUCallSiteDIE(DIE,
                     with_attr("DW_AT_low_pc"),
                     with_attr("DW_AT_abstract_origin")):
    tag = 'DW_TAG_GNU_call_site'

    def get_callee(self):
        try:
            return SubProgramDIE(self.get_DIE_from_attribute("DW_AT_abstract_origin"))
        except KeyError:
            return None
