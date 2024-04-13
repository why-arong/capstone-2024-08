import 'package:capstone/screen/script/script_list.dart';
import 'package:flutter/material.dart';
import 'package:capstone/constants/color.dart' as colors;

class ScriptTabs extends StatefulWidget{
  const ScriptTabs({Key? key}) : super(key: key);

  @override    
  ScriptTabsState createState() => ScriptTabsState();
}

class ScriptTabsState extends State<ScriptTabs> with SingleTickerProviderStateMixin {   
  late TabController _tabController;

  @override    
  void initState(){        
    super.initState();              
    _tabController = TabController(vsync: this, length: 2);    
  }        

  @override    
  void dispose(){        
    _tabController.dispose();        
    super.dispose();    
  }    
  
  final List<Tab> _tabs = [
    const Tab(
      child: Text(
        'News',
        semanticsLabel: 'News',
        style: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w800
        )
      ),
    ),
    const Tab(
      child: Text(
        'User',
        semanticsLabel: 'User',
        style: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w800
        )
      )
    )
  ];

  PreferredSize scriptTapBar() {
    return PreferredSize(
      preferredSize: const Size.fromHeight(kToolbarHeight),
      child: AppBar(
        backgroundColor: colors.bgrDarkColor,
        bottom: TabBar(
          controller: _tabController,
          labelColor: colors.exampleScriptColor,
          unselectedLabelColor: colors.blockColor,
          dividerColor: colors.bgrDarkColor,
          indicatorColor: Colors.transparent,
          isScrollable: true,
          tabAlignment: TabAlignment.start, 
          labelPadding: const EdgeInsets.only(left: 20),
          tabs: _tabs.map((label) => Container(
              child: label,
          )).toList(),
        ))
    );
  }        


  @override
  Widget build(BuildContext context){
    return Scaffold(
      appBar: scriptTapBar(),
      body: TabBarView(
          controller: _tabController,
          children: [
            for(int idx=0; idx<_tabController.length; idx++)
              ScriptList(index: idx),
      ])
    );
  }
}