import Lean
import Lean.Util.FoldConsts

open Lean
open Std (HashSet)

namespace HeytingLean.Tools

def outputPath : System.FilePath :=
  System.FilePath.mk "../blueprint/build/lean_deps.json"

def targetsPath : System.FilePath :=
  System.FilePath.mk "../blueprint/build/lean_targets.json"

def oleanRoot : System.FilePath :=
  System.FilePath.mk "../lean/.lake/build/lib/lean"

def moduleNameFromPath (root path : System.FilePath) : Option Name :=
  let rootStr := root.toString
  let clean := (path.withExtension "").toString
  if !clean.startsWith rootStr then
    none
  else
    let trimmed := clean.drop (rootStr.length + 1)
    let sep := String.singleton System.FilePath.pathSeparator
    let segments := trimmed.splitOn sep
    let name := segments.foldl
      (fun acc seg =>
        let part := seg.trim
        if part.isEmpty then acc else Name.mkStr acc part)
      Name.anonymous
    if name == Name.anonymous then none else some name

partial def collectModuleNames (dir : System.FilePath) (root : System.FilePath) : IO (Array Name) := do
  let mut acc : Array Name := #[]
  for entry in ← dir.readDir do
    let path := entry.path
    let info ← path.metadata
    match info.type with
    | IO.FS.FileType.dir =>
        acc := acc ++ (← collectModuleNames path root)
    | IO.FS.FileType.file =>
        if path.extension == some "olean" then
          if let some name := moduleNameFromPath root path then
            acc := acc.push name
    | _ => pure ()
  return acc

def parseName (s : String) : Name :=
  s.trim.splitOn "."
    |>.foldl (init := Name.anonymous) fun acc piece =>
      let segment := piece.trim
      if segment.isEmpty then acc else Name.mkStr acc segment

def jsonArrayOfStrings (value : Json) : Except String (Array String) := do
  match value with
  | Json.arr arr =>
      arr.foldlM
        (fun acc item => do
          let str ← item.getStr?
          pure <| acc.push str)
        #[]
  | _ => .error "expected JSON array"

def loadTargets (path : System.FilePath) : IO (Array Name × Array Name) := do
  unless ← path.pathExists do
    throw <| IO.userError s!"Target file not found: {path}"
  let raw ← IO.FS.readFile path
  let json ←
    match Json.parse raw with
    | Except.ok j => pure j
    | Except.error err => throw <| IO.userError err
  let modulesVal ←
    match json.getObjVal? "modules" with
    | Except.ok v => pure v
    | Except.error err => throw <| IO.userError err
  let constantsVal ←
    match json.getObjVal? "constants" with
    | Except.ok v => pure v
    | Except.error err => throw <| IO.userError err
  let modulesStrs ←
    match jsonArrayOfStrings modulesVal with
    | Except.ok arr => pure arr
    | Except.error err => throw <| IO.userError err
  let constantsStrs ←
    match jsonArrayOfStrings constantsVal with
    | Except.ok arr => pure arr
    | Except.error err => throw <| IO.userError err
  let modules := modulesStrs.map parseName
  let constants := constantsStrs.map parseName
  return (modules, constants)

def classifyKind (info : ConstantInfo) : String :=
  match info with
  | ConstantInfo.defnInfo _ => "definition"
  | ConstantInfo.thmInfo _ => "theorem"
  | ConstantInfo.axiomInfo _ => "axiom"
  | ConstantInfo.ctorInfo _ => "constructor"
  | ConstantInfo.inductInfo _ => "inductive"
  | ConstantInfo.quotInfo _ => "quot"
  | ConstantInfo.recInfo _ => "recursor"
  | ConstantInfo.opaqueInfo _ => "opaque"

def collectConstNames (e : Expr) : Array Name :=
  e.foldConsts (init := #[]) fun n acc => acc.push n

def dedup (names : Array Name) : Array Name :=
  let (_, acc) :=
    names.foldl
      (fun (state : HashSet Name × Array Name) name =>
        let (seen, out) := state
        if seen.contains name then
          (seen, out)
        else
          (seen.insert name, out.push name))
      (HashSet.empty, #[])
  acc

def buildEntry (env : Environment) (name : Name) : IO Json := do
  match env.find? name with
  | none =>
      pure <|
        Json.mkObj
          [("lean", Json.str name.toString), ("found", Json.bool false)]
  | some info =>
      let depsType := collectConstNames info.type
      let depsValue :=
        match info.value? with
        | none => (#[] : Array Name)
        | some val => collectConstNames val
      let deps :=
        (dedup <| depsType ++ depsValue).filter fun dep => dep ≠ name
      let doc? ← findDocString? env name
      let kind := classifyKind info
      pure <|
        Json.mkObj
          [ ("lean", Json.str name.toString)
          , ("found", Json.bool true)
          , ("kind", Json.str kind)
          , ("doc",
              match doc? with
              | some doc => Json.str doc
              | none => Json.null)
          , ("deps", Json.arr <| deps.map (fun dep => Json.str dep.toString))
          ]

def ensureParentDir (path : System.FilePath) : IO Unit := do
  match path.parent with
  | some dir => IO.FS.createDirAll dir
  | none => pure ()

unsafe def run : IO Unit := do
  Lean.enableInitializersExecution
  let sysroot ← Lean.findSysroot
  Lean.initSearchPath sysroot
  let (targetModules, declsRaw) ← loadTargets targetsPath
  let scannedModules ← collectModuleNames oleanRoot oleanRoot
  let allModules := targetModules ++ scannedModules
  let moduleImports : Array Import :=
    (dedup allModules).filterMap (fun name =>
      if name.getRoot == `HeytingLean then
        some ({ module := name } : Import)
      else
        none)
  let decls := dedup declsRaw
  let opts : Options := {}
  let env ← importModules (loadExts := true) moduleImports opts
  let mut entries : Array Json := #[]
  for name in decls do
    let entry ← buildEntry env name
    entries := entries.push entry
  ensureParentDir outputPath
  let json := Json.mkObj [("constants", Json.arr entries)]
  IO.FS.writeFile outputPath (json.pretty 2)
  IO.println s!"[GenerateDepGraph] wrote {entries.size} entries to {outputPath}"

end HeytingLean.Tools

unsafe def main : IO Unit := HeytingLean.Tools.run
