import HeytingLean
import HeytingLean.Runtime.BridgeSuite
import HeytingLean.Docs.BooleanDemo

open HeytingLean.Runtime

private def describeToggle (name : String) (enabled : Bool) : String :=
  s!"  - {name}: {(if enabled then "enabled" else "disabled")}"

def main : IO Unit := do
  -- Initialise the enriched runtime suite using the documented default nucleus.
  let _ := bridgeSuite (α := Set Unit) (R := HeytingLean.Docs.idNucleus)
  let flags := bridgeFlags
  IO.println "HeytingLean runtime bridge suite loaded with enriched carriers."
  IO.println (describeToggle "tensor intensity" flags.useTensorIntensity)
  IO.println (describeToggle "graph Alexandroff" flags.useGraphAlexandroff)
  IO.println (describeToggle "Clifford projector" flags.useCliffordProjector)
