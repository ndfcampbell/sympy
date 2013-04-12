%(subroutine_header)s

%(use_statements)s

implicit none

! ===================== !
! Argument Declarations !
! ===================== !

%(argument_declarations)s

! ===================== !
! Variable Declarations !
! ===================== !

%(variable_declarations)s

begin interface
%(function_interfaces)s
end interface

! ======================== !
! Variable Initializations !
! ======================== !
%(variable_initializations)s

! ========== !
! Statements !
! ========== !
%(statements)s

! ======================= !
! Variable Deconstruction !
! ======================= !
%(variable_destructions)s

return

%(footer)s

%(function_definitions)s

