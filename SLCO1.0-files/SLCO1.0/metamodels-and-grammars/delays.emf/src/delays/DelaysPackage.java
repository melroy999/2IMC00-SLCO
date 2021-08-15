/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package delays;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains accessors for the meta objects to represent
 * <ul>
 *   <li>each class,</li>
 *   <li>each feature of each class,</li>
 *   <li>each enum,</li>
 *   <li>and each data type</li>
 * </ul>
 * <!-- end-user-doc -->
 * @see delays.DelaysFactory
 * @model kind="package"
 * @generated
 */
public interface DelaysPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "delays";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "delays";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "delays";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	DelaysPackage eINSTANCE = delays.impl.DelaysPackageImpl.init();

	/**
	 * The meta object id for the '{@link delays.impl.TransitionListImpl <em>Transition List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see delays.impl.TransitionListImpl
	 * @see delays.impl.DelaysPackageImpl#getTransitionList()
	 * @generated
	 */
	int TRANSITION_LIST = 0;

	/**
	 * The feature id for the '<em><b>Transitions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION_LIST__TRANSITIONS = 0;

	/**
	 * The number of structural features of the '<em>Transition List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link delays.impl.TransitionImpl <em>Transition</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see delays.impl.TransitionImpl
	 * @see delays.impl.DelaysPackageImpl#getTransition()
	 * @generated
	 */
	int TRANSITION = 1;

	/**
	 * The feature id for the '<em><b>Delay</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION__DELAY = 0;

	/**
	 * The feature id for the '<em><b>Transition</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION__TRANSITION = 1;

	/**
	 * The number of structural features of the '<em>Transition</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION_FEATURE_COUNT = 2;


	/**
	 * Returns the meta object for class '{@link delays.TransitionList <em>Transition List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Transition List</em>'.
	 * @see delays.TransitionList
	 * @generated
	 */
	EClass getTransitionList();

	/**
	 * Returns the meta object for the containment reference list '{@link delays.TransitionList#getTransitions <em>Transitions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Transitions</em>'.
	 * @see delays.TransitionList#getTransitions()
	 * @see #getTransitionList()
	 * @generated
	 */
	EReference getTransitionList_Transitions();

	/**
	 * Returns the meta object for class '{@link delays.Transition <em>Transition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Transition</em>'.
	 * @see delays.Transition
	 * @generated
	 */
	EClass getTransition();

	/**
	 * Returns the meta object for the attribute '{@link delays.Transition#getDelay <em>Delay</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Delay</em>'.
	 * @see delays.Transition#getDelay()
	 * @see #getTransition()
	 * @generated
	 */
	EAttribute getTransition_Delay();

	/**
	 * Returns the meta object for the reference '{@link delays.Transition#getTransition <em>Transition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Transition</em>'.
	 * @see delays.Transition#getTransition()
	 * @see #getTransition()
	 * @generated
	 */
	EReference getTransition_Transition();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	DelaysFactory getDelaysFactory();

	/**
	 * <!-- begin-user-doc -->
	 * Defines literals for the meta objects that represent
	 * <ul>
	 *   <li>each class,</li>
	 *   <li>each feature of each class,</li>
	 *   <li>each enum,</li>
	 *   <li>and each data type</li>
	 * </ul>
	 * <!-- end-user-doc -->
	 * @generated
	 */
	interface Literals {
		/**
		 * The meta object literal for the '{@link delays.impl.TransitionListImpl <em>Transition List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see delays.impl.TransitionListImpl
		 * @see delays.impl.DelaysPackageImpl#getTransitionList()
		 * @generated
		 */
		EClass TRANSITION_LIST = eINSTANCE.getTransitionList();

		/**
		 * The meta object literal for the '<em><b>Transitions</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TRANSITION_LIST__TRANSITIONS = eINSTANCE.getTransitionList_Transitions();

		/**
		 * The meta object literal for the '{@link delays.impl.TransitionImpl <em>Transition</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see delays.impl.TransitionImpl
		 * @see delays.impl.DelaysPackageImpl#getTransition()
		 * @generated
		 */
		EClass TRANSITION = eINSTANCE.getTransition();

		/**
		 * The meta object literal for the '<em><b>Delay</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute TRANSITION__DELAY = eINSTANCE.getTransition_Delay();

		/**
		 * The meta object literal for the '<em><b>Transition</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TRANSITION__TRANSITION = eINSTANCE.getTransition_Transition();

	}

} //DelaysPackage
