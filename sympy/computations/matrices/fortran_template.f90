%(function_definitions)s

%(subroutine_header)s

%(use_statements)s

implicit none

! args? what does this mean?

begin interface
%(function_interfaces)s
end interface

! ===================== !
! Variable Declarations !
! ===================== !

%(variable_declarations)s

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
