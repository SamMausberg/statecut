import StateCut
import Lean.Util.CollectAxioms

open Lean Elab Command

/-! Audit every compiled constant in the application namespace, including
compiler-generated auxiliary declarations. The allowed names are Lean's
standard logical foundations; all other proof dependencies fail the command.
This checks proof dependencies, not the correctness of external executables. -/

run_cmd do
  let env ← getEnv
  let constants := env.constants.toList.filter fun (name, _) =>
    (`StateCut).isPrefixOf name
  let constants := constants.mergeSort fun a b => a.1.toString ≤ b.1.toString
  if constants.isEmpty then
    throwError "No StateCut declarations found"
  for (name, info) in constants do
    if info.isUnsafe then
      -- Lean emits extraction-stage runtime auxiliaries for safe executable
      -- definitions. These are outside the logical kernel-proof inventory.
      if (name.toString.splitOn "._cstage").length == 1 then
        throwError "Unexpected unchecked declaration: {name}"
      logInfo m!"STATECUT_RUNTIME_AUX {toJson name.toString |>.compress}"
      continue
    let deps ← collectAxioms name
    let forbidden := deps.filter fun dep =>
      dep != `propext && dep != `Classical.choice && dep != `Quot.sound
    unless forbidden.isEmpty do
      throwError "Unexpected proof dependencies for {name}: {forbidden}"
    let kind := match info with
      | .thmInfo _ => "theorem"
      | .defnInfo _ => "definition"
      | .axiomInfo _ => "axiom"
      | _ => "other"
    let record := Json.mkObj [
      ("name", toJson name.toString),
      ("kind", toJson kind),
      ("dependencies", toJson (deps.map Name.toString))]
    logInfo m!"STATECUT_AUDIT {record.compress}"
