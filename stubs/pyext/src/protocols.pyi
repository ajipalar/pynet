from .typedefs import PRNGKeyArray as PRNGKeyArray, T as T, T_co as T_co, T_contra as T_contra

class SupportsPDF:
    def pdf(x: T_co) -> float: ...

class SupportsLogPDF:
    def lpdf(x: T_co) -> float: ...

class SupportsRV:
    def rv(key: PRNGKeyArray) -> T_co: ...
