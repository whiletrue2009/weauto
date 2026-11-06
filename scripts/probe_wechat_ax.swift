import ApplicationServices
import CoreGraphics
import Foundation

struct Options {
    var pid: pid_t = 0
    var titleFilter = ""
    var maxDepth = 12
    var maxNodes = 2000
    var showValues = false
    var showAttributes = false
    var showApplication = false
    var setEnhancedUserInterface: Bool?
    var raiseWindow = false
    var hitPoint: CGPoint?
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func parseOptions() -> Options {
    var options = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    while !args.isEmpty {
        let arg = args.removeFirst()
        func value() -> String {
            guard !args.isEmpty else { fail("Missing value for \(arg)") }
            return args.removeFirst()
        }
        switch arg {
        case "--pid": options.pid = pid_t(Int32(value()) ?? 0)
        case "--title": options.titleFilter = value()
        case "--max-depth": options.maxDepth = Int(value()) ?? options.maxDepth
        case "--max-nodes": options.maxNodes = Int(value()) ?? options.maxNodes
        case "--values": options.showValues = true
        case "--attributes": options.showAttributes = true
        case "--application": options.showApplication = true
        case "--set-enhanced": options.setEnhancedUserInterface = (value() == "1")
        case "--raise": options.raiseWindow = true
        case "--hit": 
            let coordinate = value().split(separator: ",").compactMap { Double($0) }
            guard coordinate.count == 2 else { fail("Hit point must be x,y") }
            options.hitPoint = CGPoint(x: coordinate[0], y: coordinate[1])
        default: fail("Unknown argument: \(arg)")
        }
    }
    return options
}

func copyAttribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    let error = AXUIElementCopyAttributeValue(element, name as CFString, &value)
    guard error == .success else { return nil }
    return value
}

func children(_ element: AXUIElement) -> [AXUIElement] {
    guard let value = copyAttribute(element, kAXChildrenAttribute as String) else { return [] }
    if let elements = value as? [AXUIElement] { return elements }
    if let array = value as? [Any] {
        return array.map { unsafeBitCast($0, to: AXUIElement.self) }
    }
    return []
}

func attributeNames(_ element: AXUIElement) -> [String] {
    var names: CFArray?
    let error = AXUIElementCopyAttributeNames(element, &names)
    guard error == .success, let names = names as? [String] else { return [] }
    return names
}

func childElements(_ element: AXUIElement) -> [(attribute: String, elements: [AXUIElement])] {
    let names = [
        kAXChildrenAttribute as String,
        "AXChildrenInNavigationOrder",
        "AXVisibleChildren",
        kAXContentsAttribute as String,
        kAXRowsAttribute as String,
        kAXTabsAttribute as String,
        "AXTitleUIElements",
        "AXSections",
    ]
    return names.map { name in
        guard let value = copyAttribute(element, name) else { return (name, []) }
        if let elements = value as? [AXUIElement] { return (name, elements) }
        if let array = value as? [Any] {
            return (name, array.map { unsafeBitCast($0, to: AXUIElement.self) })
        }
        if name == "AXSections",
           let sections = value as? [[String: Any]] {
            let sectionObjects = sections.compactMap { section -> AXUIElement? in
                guard let object = section["SectionObject"] as? CFTypeRef,
                      CFGetTypeID(object) == AXUIElementGetTypeID() else { return nil }
                return unsafeBitCast(object, to: AXUIElement.self)
            }
            if !sectionObjects.isEmpty { return (name, sectionObjects) }
        }
        return (name, [])
    }
}

func textValue(_ element: AXUIElement, _ attribute: String) -> String {
    guard let value = copyAttribute(element, attribute) else { return "" }
    if let text = value as? String { return text }
    return String(describing: value)
}

func frame(_ element: AXUIElement) -> String {
    var x = ""
    var y = ""
    var width = ""
    var height = ""
    if let positionValue = copyAttribute(element, kAXPositionAttribute as String) {
        let axValue = unsafeBitCast(positionValue, to: AXValue.self)
        var point = CGPoint.zero
        if AXValueGetValue(axValue, .cgPoint, &point) {
            x = String(format: "%.0f", point.x)
            y = String(format: "%.0f", point.y)
        }
    }
    if let sizeValue = copyAttribute(element, kAXSizeAttribute as String) {
        let axValue = unsafeBitCast(sizeValue, to: AXValue.self)
        var size = CGSize.zero
        if AXValueGetValue(axValue, .cgSize, &size) {
            width = String(format: "%.0f", size.width)
            height = String(format: "%.0f", size.height)
        }
    }
    return "x=\(x) y=\(y) w=\(width) h=\(height)"
}

func briefDescription(_ element: AXUIElement, path: String) {
    let role = textValue(element, kAXRoleAttribute as String)
    var fields: [String] = [
        "path=\(path)",
        "role=\(role)",
        frame(element),
    ]
    for (attribute, label) in [
        (kAXSubroleAttribute as String, "subrole"),
        (kAXTitleAttribute as String, "title"),
        (kAXIdentifierAttribute as String, "identifier"),
        (kAXDescriptionAttribute as String, "description"),
        (kAXValueAttribute as String, "value"),
    ] {
        let text = textValue(element, attribute)
        if !text.isEmpty {
            let shown = options.showValues
                ? text.replacingOccurrences(of: "\n", with: "\\n")
                : "<length=\(text.count)>"
            fields.append("\(label)=\(shown)")
        }
    }
    print("element " + fields.joined(separator: " "))
}

func inspectAttributes(_ element: AXUIElement, path: String) {
    let names = attributeNames(element)
    print("attributes[\(path)] \(names.joined(separator: ","))")
    for name in names {
        guard options.showValues,
              name != kAXChildrenAttribute as String,
              name != "AXChildrenInNavigationOrder",
              name != "AXVisibleChildren",
              name != kAXContentsAttribute as String,
              name != kAXRowsAttribute as String,
              name != kAXTabsAttribute as String,
              name != "AXTitleUIElements",
              name != "AXSections",
              name != kAXParentAttribute as String,
              name != kAXWindowAttribute as String else { continue }
        guard let value = copyAttribute(element, name) else {
            print("attribute[\(path)] \(name)=<unreadable>")
            continue
        }
        var description = String(describing: value)
            .replacingOccurrences(of: "\n", with: "\\n")
        if description.count > 500 {
            description = String(description.prefix(500)) + "...<truncated>"
        }
        print("attribute[\(path)] \(name)=\(description)")
    }
}

func walkAllAttributes(_ element: AXUIElement, path: String, depth: Int) {
    guard depth <= options.maxDepth, visited < options.maxNodes else { return }
    visited += 1
    let role = textValue(element, kAXRoleAttribute as String)
    roleCounts[role, default: 0] += 1
    briefDescription(element, path: path)
    if options.showAttributes {
        inspectAttributes(element, path: path)
    }

    var offset = 0
    for childList in childElements(element) where !childList.elements.isEmpty {
        print("children[\(path)] \(childList.attribute)=\(childList.elements.count)")
        for child in childList.elements {
            walkAllAttributes(child, path: "\(path)/\(childList.attribute)[\(offset)]", depth: depth + 1)
            offset += 1
        }
    }
}

let options = parseOptions()
let pid = options.pid
guard pid > 0 else {
    fail("Usage: probe_wechat_ax --pid PID [--title TEXT] [--max-depth N] [--max-nodes N] [--values] [--attributes] [--hit x,y]")
}

let application = AXUIElementCreateApplication(pid)
AXUIElementSetMessagingTimeout(application, 1.0)
guard let windowValue = copyAttribute(application, kAXWindowsAttribute as String),
      let windows = windowValue as? [AXUIElement] else {
    fail("Unable to read AXWindows for pid \(pid)")
}

print("windows=\(windows.count)")
if let enhanced = options.setEnhancedUserInterface {
    let error = AXUIElementSetAttributeValue(
        application,
        "AXEnhancedUserInterface" as CFString,
        enhanced as CFBoolean
    )
    print("set AXEnhancedUserInterface=\(enhanced) error=\(error.rawValue)")
}
if options.showApplication {
    inspectAttributes(application, path: "application")
}
var selectedWindows: [(index: Int, element: AXUIElement)] = []
for (index, window) in windows.enumerated() {
    let role = textValue(window, kAXRoleAttribute as String)
    let title = textValue(window, kAXTitleAttribute as String)
    let subrole = textValue(window, kAXSubroleAttribute as String)
    print("window[\(index)] role=\(role) subrole=\(subrole) title=\(title) \(frame(window)) children=\(children(window).count)")
    if options.titleFilter.isEmpty || title.localizedCaseInsensitiveContains(options.titleFilter) {
        selectedWindows.append((index, window))
    }
}

guard !selectedWindows.isEmpty else { fail("No window matches title filter: \(options.titleFilter)") }

if options.raiseWindow {
    for window in selectedWindows {
        let error = AXUIElementPerformAction(window.element, kAXRaiseAction as CFString)
        print("raise window[\(window.index)] error=\(error.rawValue)")
    }
    Thread.sleep(forTimeInterval: 1.0)
}

var visited = 0
var textNodes = 0
var textCharacters = 0
var roleCounts: [String: Int] = [:]

func walk(_ element: AXUIElement, path: String, depth: Int) {
    guard visited < options.maxNodes, depth <= options.maxDepth else { return }
    visited += 1
    let role = textValue(element, kAXRoleAttribute as String)
    roleCounts[role, default: 0] += 1

    var fields: [String] = []
    fields.append("role=\(role)")
    fields.append(frame(element))

    for (attribute, label) in [
        (kAXTitleAttribute as String, "title"),
        (kAXIdentifierAttribute as String, "identifier"),
        (kAXDescriptionAttribute as String, "description"),
        (kAXValueAttribute as String, "value"),
    ] {
        let text = textValue(element, attribute)
        if !text.isEmpty {
            if attribute == kAXValueAttribute as String {
                textNodes += 1
                textCharacters += text.count
                if options.showValues {
                    fields.append("\(label)=\(text.replacingOccurrences(of: "\n", with: "\\n"))")
                } else {
                    fields.append("\(label).length=\(text.count)")
                }
            } else {
                fields.append("\(label)=\(text.replacingOccurrences(of: "\n", with: "\\n"))")
            }
        }
    }
    print("node[\(visited)] path=\(path) " + fields.joined(separator: " "))

    for (offset, child) in children(element).enumerated() {
        walk(child, path: "\(path)/\(offset)", depth: depth + 1)
    }
}

func inspectHitChain(_ startPoint: CGPoint) {
    let systemWide = AXUIElementCreateSystemWide()
    AXUIElementSetMessagingTimeout(systemWide, 1.0)
    for context in [(name: "system", element: systemWide), (name: "application", element: application)] {
        var hitElement: AXUIElement?
        let error = AXUIElementCopyElementAtPosition(
            context.element,
            Float(startPoint.x),
            Float(startPoint.y),
            &hitElement
        )
        guard error == .success, let hitElement = hitElement else {
            print("hit context=\(context.name) error=\(error.rawValue) point=\(startPoint.x),\(startPoint.y)")
            continue
        }

        print("hit context=\(context.name) point=\(startPoint.x),\(startPoint.y)")
        var current: AXUIElement? = hitElement
        var depth = 0
        while let element = current, depth <= options.maxDepth {
            let path = "hit/\(context.name)/parent[\(depth)]"
            briefDescription(element, path: path)
            if options.showAttributes {
                inspectAttributes(element, path: path)
            }
            depth += 1
            if let parentValue = copyAttribute(element, kAXParentAttribute as String) {
                current = unsafeBitCast(parentValue, to: AXUIElement.self)
            } else {
                current = nil
            }
        }
    }
}

for window in selectedWindows {
    print("=== BEGIN window[\(window.index)] ===")
    walkAllAttributes(window.element, path: "window[\(window.index)]", depth: 1)
    print("=== END window[\(window.index)] ===")
}

if let hitPoint = options.hitPoint {
    print("=== BEGIN hit chain ===")
    inspectHitChain(hitPoint)
    print("=== END hit chain ===")
}

print("summary visited=\(visited) textNodes=\(textNodes) textCharacters=\(textCharacters)")
for (role, count) in roleCounts.sorted(by: { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }) {
    print("roleCount \(role)=\(count)")
}
