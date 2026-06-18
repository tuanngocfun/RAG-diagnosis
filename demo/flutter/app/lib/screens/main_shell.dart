import 'package:flutter/material.dart';

import '../services/backend_client.dart';
import 'consult_screen.dart';
import 'gpu_assistant_screen.dart';

class MainShell extends StatefulWidget {
  const MainShell({
    required this.client,
    super.key,
  });

  final BackendClient client;

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    final List<_DemoDestination> destinations = <_DemoDestination>[
      _DemoDestination(
        label: 'Consult Gate',
        icon: Icons.fact_check_outlined,
        selectedIcon: Icons.fact_check,
        page: ConsultScreen(client: widget.client),
      ),
      _DemoDestination(
        label: 'GPU Assistant',
        icon: Icons.smart_toy_outlined,
        selectedIcon: Icons.smart_toy,
        page: GpuAssistantScreen(client: widget.client),
      ),
    ];

    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        final bool wide = constraints.maxWidth >= 760;
        final Widget selectedPage = destinations[_selectedIndex].page;

        return Scaffold(
          appBar: AppBar(
            title: Text(destinations[_selectedIndex].label),
          ),
          body: wide
              ? Row(
                  children: <Widget>[
                    NavigationRail(
                      selectedIndex: _selectedIndex,
                      onDestinationSelected: (int index) {
                        setState(() {
                          _selectedIndex = index;
                        });
                      },
                      labelType: NavigationRailLabelType.all,
                      destinations: destinations
                          .map(
                            (_DemoDestination item) =>
                                NavigationRailDestination(
                              icon: Icon(item.icon),
                              selectedIcon: Icon(item.selectedIcon),
                              label: Text(item.label),
                            ),
                          )
                          .toList(),
                    ),
                    const VerticalDivider(width: 1),
                    Expanded(child: selectedPage),
                  ],
                )
              : selectedPage,
          bottomNavigationBar: wide
              ? null
              : NavigationBar(
                  selectedIndex: _selectedIndex,
                  onDestinationSelected: (int index) {
                    setState(() {
                      _selectedIndex = index;
                    });
                  },
                  destinations: destinations
                      .map(
                        (_DemoDestination item) => NavigationDestination(
                          icon: Icon(item.icon),
                          selectedIcon: Icon(item.selectedIcon),
                          label: item.label,
                        ),
                      )
                      .toList(),
                ),
        );
      },
    );
  }
}

class _DemoDestination {
  const _DemoDestination({
    required this.label,
    required this.icon,
    required this.selectedIcon,
    required this.page,
  });

  final String label;
  final IconData icon;
  final IconData selectedIcon;
  final Widget page;
}
