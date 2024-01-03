use pyo3::prelude::*;

#[pyfunction]
fn rust_function() -> PyResult<()> {
    // add later
    Ok(())
}

#[pymodule]
fn rust_ml_component(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_function, m)?)?;
    Ok(())
}
